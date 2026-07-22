"""
Expected-loss decision engine — the pricing layer.

Optimizes expected PESO LOSS, not classification accuracy. Escalation to a human
is a cost (~₱19/case), not a free safety net; it is only worth paying when the
expected fraud loss it prevents exceeds that cost. For a low-value claim the
expected fraud loss is tiny, so auto-approving (and eating rare fraud) is provably
cheaper than reviewing — which is why the decision must be VALUE-SENSITIVE.

Terminology (per the Shopee case spec — note the redefinition):
  positive        = INVALID proof
  false negative  = invalid proof classified valid  = bad approval  (pays a fraudster)
  false positive  = valid proof classified invalid  = bad rejection
  escalation      = a routing action, NOT a false positive/negative

Assumption provenance is tagged inline:
  [FACT]                  publicly reported / directly observed
  [DERIVED]               calculated from facts
  [ILLUSTRATIVE]          scenario assumption
  [SHOPEE INPUT REQUIRED] cannot be known from public data — MUST be calibrated

CALIBRATION GUARD: `p_invalid` must be a genuine probability. Our model's
credibility score is NOT calibrated, so we never pass it in raw. The pipeline
supplies `p_invalid` from configurable per-decision-bucket base rates
(`EconomicConfig.bucket_p_invalid`), themselves SHOPEE-INPUT-REQUIRED placeholders.
Any case with a missing price or an out-of-range p_invalid is routed to a human.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

# Review-cost sensitivity cases (fully loaded operating cost per escalation, PHP).
# ₱19 [DERIVED] from the fact sheet's reviewer salary proxy + statutory loading +
# overhead + dataset-weighted review time. ₱10 / ₱40 are the low / high sensitivity
# bounds requested by the spec.
REVIEW_COST_LOW = 10.0
REVIEW_COST_NORMAL = 19.0
REVIEW_COST_HIGH = 40.0

# Route labels this engine emits (approve/escalate only — auto-REJECT is owned by
# the evidence layer + an explicit config flag, never by this economic function).
ROUTE_AUTO_APPROVE = "auto_approve"
ROUTE_ESCALATE = "escalate"


@dataclass
class EconomicConfig:
    """Every economic input, all configurable. Defaults = the Normal scenario."""

    # Fraction of the claim value Shopee actually eats on a bad approval.
    shopee_net_loss_fraction: float = 0.50        # [ILLUSTRATIVE]
    # Fully loaded human-review cost per escalation.
    review_cost_php: float = REVIEW_COST_NORMAL   # [DERIVED]
    # Cost attributed to the delay a human review adds (SLA / buyer-experience).
    delay_cost_php: float = 1.0                   # [ILLUSTRATIVE]
    # P(human correctly flags an invalid case they review).
    human_invalid_detection_rate: float = 0.90    # [SHOPEE INPUT REQUIRED]
    # P(human wrongly rejects a valid case they review).
    human_false_reject_rate: float = 0.02         # [SHOPEE INPUT REQUIRED]
    # Cost of wrongly rejecting a valid claim (buyer harm, appeal, churn, LTV).
    wrong_rejection_cost_php: float = 100.0       # [SHOPEE INPUT REQUIRED]

    # Per-decision-bucket invalid base rates. The model's credibility is NOT a
    # calibrated probability, so instead of using it directly we assign a
    # conservative invalid-rate to each decision bucket. [SHOPEE INPUT REQUIRED]
    # — calibrate these against labeled outcomes from the human swipe queue.
    bucket_p_invalid: dict = field(default_factory=lambda: {
        "approve": 0.05,     # cases the evidence layer would auto-approve
        "escalate": 0.30,    # cases the evidence layer is uncertain about
    })

    # Auto-rejection stays OFF unless explicitly enabled. Shopee guidance: keep it
    # disabled until the cost of wrongly rejecting a valid claim is known. Default
    # True here preserves the current fraud-veto behavior; flip to False to make
    # every model reject an escalation instead.
    enable_auto_reject: bool = True

    # --- scenario presets ----------------------------------------------------

    @classmethod
    def normal(cls) -> "EconomicConfig":
        return cls()

    @classmethod
    def bull(cls) -> "EconomicConfig":
        return cls(shopee_net_loss_fraction=1.00, review_cost_php=REVIEW_COST_LOW,
                   delay_cost_php=0.50, human_invalid_detection_rate=0.95,
                   human_false_reject_rate=0.01, wrong_rejection_cost_php=50.0)

    @classmethod
    def bear(cls) -> "EconomicConfig":
        return cls(shopee_net_loss_fraction=0.25, review_cost_php=REVIEW_COST_HIGH,
                   delay_cost_php=3.0, human_invalid_detection_rate=0.75,
                   human_false_reject_rate=0.05, wrong_rejection_cost_php=200.0)


@dataclass
class EscalationDecision:
    """Full, auditable trace of one auto-approve-vs-escalate decision."""
    route: str                              # ROUTE_AUTO_APPROVE | ROUTE_ESCALATE
    reason: str                             # plain-language peso justification
    claim_value_php: Optional[float]
    p_invalid: Optional[float]
    net_loss_exposure: Optional[float]
    expected_loss_auto_approve: Optional[float]
    expected_loss_escalate: Optional[float]
    p_threshold_full: Optional[float]       # break-even p(invalid), full formula
    p_threshold_simple: Optional[float]     # review_cost / net_loss_exposure

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Break-even helpers
# ---------------------------------------------------------------------------
def break_even_p_full(net_loss_exposure: float, cfg: EconomicConfig) -> Optional[float]:
    """Invalid probability at which escalation and auto-approval cost the same.

    p* = (review + delay + false_reject × wrong_reject_cost)
         / (detection × net_loss_exposure + false_reject × wrong_reject_cost)
    """
    denom = (cfg.human_invalid_detection_rate * net_loss_exposure
             + cfg.human_false_reject_rate * cfg.wrong_rejection_cost_php)
    if denom <= 0:
        return None
    numer = (cfg.review_cost_php + cfg.delay_cost_php
             + cfg.human_false_reject_rate * cfg.wrong_rejection_cost_php)
    return numer / denom


def break_even_p_simple(net_loss_exposure: float, cfg: EconomicConfig) -> Optional[float]:
    """Simplified break-even ignoring human error + delay: review_cost / exposure."""
    if net_loss_exposure <= 0:
        return None
    return cfg.review_cost_php / net_loss_exposure


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------
def expected_loss_decision(claim_value_php: Optional[float],
                           p_invalid: Optional[float],
                           cfg: Optional[EconomicConfig] = None) -> EscalationDecision:
    """Decide auto-approve vs escalate by expected peso loss.

    expected_loss_auto_approve = p_invalid × net_loss_exposure
    expected_loss_escalate     = review + delay
                                 + p_invalid × (1 - detection) × net_loss_exposure
                                 + (1 - p_invalid) × false_reject × wrong_reject_cost
    Escalate when expected_loss_escalate < expected_loss_auto_approve.

    Fail-safe routing (spec): a case we cannot price or whose p_invalid is not a
    valid calibrated probability goes to a human, never to silent auto-approval.
    """
    cfg = cfg or EconomicConfig.normal()

    def _escalate(reason, **extra) -> EscalationDecision:
        base = dict(route=ROUTE_ESCALATE, reason=reason,
                    claim_value_php=claim_value_php, p_invalid=p_invalid,
                    net_loss_exposure=None, expected_loss_auto_approve=None,
                    expected_loss_escalate=None, p_threshold_full=None,
                    p_threshold_simple=None)
        base.update(extra)
        return EscalationDecision(**base)

    # --- guards: never auto-approve what we cannot safely evaluate ------------
    if claim_value_php is None:
        return _escalate("missing_price: claim value unknown, routed to human review")
    if claim_value_php < 0:
        return _escalate("invalid_price: negative claim value, routed to human review")
    if p_invalid is None or not (0.0 <= p_invalid <= 1.0):
        return _escalate("uncalibrated_probability: p_invalid missing or out of "
                         "[0,1], routed to human review")

    net_loss_exposure = claim_value_php * cfg.shopee_net_loss_fraction
    p_full = break_even_p_full(net_loss_exposure, cfg)
    p_simple = break_even_p_simple(net_loss_exposure, cfg)

    # Zero exposure: nothing to lose on a bad approval, so a review can only add
    # cost (and risk a wrongful rejection). Auto-approve.
    if net_loss_exposure <= 0:
        return EscalationDecision(
            route=ROUTE_AUTO_APPROVE,
            reason=f"zero_exposure: net loss exposure is ₱0.00, so review cost "
                   f"₱{cfg.review_cost_php:.2f} cannot be justified — auto-approve",
            claim_value_php=claim_value_php, p_invalid=p_invalid,
            net_loss_exposure=net_loss_exposure,
            expected_loss_auto_approve=0.0,
            expected_loss_escalate=cfg.review_cost_php + cfg.delay_cost_php,
            p_threshold_full=p_full, p_threshold_simple=p_simple)

    el_approve = p_invalid * net_loss_exposure
    el_escalate = (cfg.review_cost_php + cfg.delay_cost_php
                   + p_invalid * (1.0 - cfg.human_invalid_detection_rate) * net_loss_exposure
                   + (1.0 - p_invalid) * cfg.human_false_reject_rate
                   * cfg.wrong_rejection_cost_php)

    # Escalate when it is STRICTLY cheaper. On an exact tie, escalate — cost
    # asymmetry favors the human when the money is indifferent.
    if el_escalate <= el_approve:
        route, verb = ROUTE_ESCALATE, "≤"
        reason = (f"escalate: expected loss of review ₱{el_escalate:.2f} {verb} "
                  f"expected loss of auto-approve ₱{el_approve:.2f} "
                  f"(claim ₱{claim_value_php:,.0f}, exposure ₱{net_loss_exposure:,.0f}, "
                  f"break-even p={p_full:.1%}, this p={p_invalid:.1%})")
    else:
        route = ROUTE_AUTO_APPROVE
        reason = (f"auto_approve: expected loss of auto-approve ₱{el_approve:.2f} < "
                  f"expected loss of review ₱{el_escalate:.2f} — the claim is too "
                  f"low-value to justify a ₱{cfg.review_cost_php:.2f} review "
                  f"(exposure ₱{net_loss_exposure:,.0f}, break-even p={p_full:.1%}, "
                  f"this p={p_invalid:.1%})")

    return EscalationDecision(
        route=route, reason=reason,
        claim_value_php=claim_value_php, p_invalid=p_invalid,
        net_loss_exposure=net_loss_exposure,
        expected_loss_auto_approve=el_approve,
        expected_loss_escalate=el_escalate,
        p_threshold_full=p_full, p_threshold_simple=p_simple)


# ---------------------------------------------------------------------------
# Scenario analysis (per 1,000 escalated cases)
# ---------------------------------------------------------------------------
# Central claim values per scenario (order-value midpoints, PHP). [ILLUSTRATIVE]
SCENARIOS = {
    "bull": {"cfg": EconomicConfig.bull, "claim_value": 1069.0, "p_invalid": 0.10},
    "normal": {"cfg": EconomicConfig.normal, "claim_value": 841.0, "p_invalid": 0.08},
    "bear": {"cfg": EconomicConfig.bear, "claim_value": 200.0, "p_invalid": 0.05},
}


def scenario_metrics(claim_value: float, p_invalid: float, cfg: EconomicConfig,
                     n: int = 1000) -> dict:
    """The 8 required outputs for one scenario, per-case and scaled to n cases."""
    net_loss = claim_value * cfg.shopee_net_loss_fraction
    loss_avoided = p_invalid * cfg.human_invalid_detection_rate * net_loss
    review = cfg.review_cost_php
    wrongful_reject = ((1.0 - p_invalid) * cfg.human_false_reject_rate
                       * cfg.wrong_rejection_cost_php)
    delay = cfg.delay_cost_php
    total_cost = review + wrongful_reject + delay
    net_benefit = loss_avoided - total_cost
    bcr = (loss_avoided / total_cost) if total_cost > 0 else float("inf")
    return {
        "net_loss_exposure": net_loss,
        "incorrect_approval_loss_avoided": loss_avoided,
        "human_review_cost": review,
        "wrongful_rejection_cost": wrongful_reject,
        "delay_cost": delay,
        "net_economic_benefit": net_benefit,
        "benefit_cost_ratio": bcr,
        "break_even_p_invalid_full": break_even_p_full(net_loss, cfg),
        "break_even_p_invalid_simple": break_even_p_simple(net_loss, cfg),
        # scaled to n escalated cases
        "per_n": {
            "n": n,
            "loss_avoided": loss_avoided * n,
            "review_cost": review * n,
            "wrongful_rejection_cost": wrongful_reject * n,
            "delay_cost": delay * n,
            "net_benefit": net_benefit * n,
        },
    }


def scenario_report(n: int = 1000) -> dict:
    """All three scenarios' metrics, keyed by name."""
    out = {}
    for name, spec in SCENARIOS.items():
        out[name] = scenario_metrics(spec["claim_value"], spec["p_invalid"],
                                     spec["cfg"](), n=n)
    return out


def _print_report(n: int = 1000) -> None:
    rep = scenario_report(n)
    rows = [
        ("net loss exposure /case", "net_loss_exposure", "₱{:,.2f}"),
        ("bad-approval loss avoided /case", "incorrect_approval_loss_avoided", "₱{:,.2f}"),
        ("human review cost /case", "human_review_cost", "₱{:,.2f}"),
        ("wrongful rejection cost /case", "wrongful_rejection_cost", "₱{:,.2f}"),
        ("delay cost /case", "delay_cost", "₱{:,.2f}"),
        ("net economic benefit /case", "net_economic_benefit", "₱{:,.2f}"),
        ("benefit-cost ratio", "benefit_cost_ratio", "{:.2f}"),
        ("break-even p(invalid) full", "break_even_p_invalid_full", "{:.2%}"),
        ("break-even p(invalid) simple", "break_even_p_invalid_simple", "{:.2%}"),
    ]
    names = ["bull", "normal", "bear"]
    print(f"{'metric':34}" + "".join(f"{nm:>16}" for nm in names))
    print("-" * (34 + 16 * len(names)))
    for label, key, fmt in rows:
        line = f"{label:34}"
        for nm in names:
            line += f"{fmt.format(rep[nm][key]):>16}"
        print(line)
    print(f"\nnet benefit per {n} escalated cases:")
    for nm in names:
        print(f"  {nm:8} ₱{rep[nm]['per_n']['net_benefit']:>14,.2f}")


if __name__ == "__main__":
    _print_report()
