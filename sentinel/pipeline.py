"""
The orchestrator — runs one case through the whole flow, in fail-fast order.

  Rung 0 (deterministic)  → terminal? stop.
  Rung 1a Authenticity    → dispositive fraud (meets the veto bar)? reject + stop.
  Rung 1b other checks    → run the rest (as they get built), then score + route.

Each short-circuit skips the paid work below it. Authenticity uses the SAME veto
threshold it would have used running in parallel, so running it first changes
cost/latency, not the standard of evidence to reject (guardrail g in the brief).

Rung 1b is not fully built yet: Completeness / Tamper / Relevance / Defender slot
in where marked. Until then a non-dispositive case is scored on Authenticity alone
(the missing checks are neutral/excluded by the combiner), so this already runs
end-to-end and every case still gets a full row + a confusion-matrix entry.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from . import scoring, db
from .contract import (
    CaseRecord, DECISION_APPROVE, DECISION_REJECT, DECISION_ESCALATE,
    DECISION_RESUBMIT,
    REASON_AUTHENTICITY_DISPOSITIVE, REASON_PASSED_PREVALIDATION,
)
from .economics import EconomicConfig, expected_loss_decision, ROUTE_AUTO_APPROVE
from .prevalidation import prevalidate, apply as apply_rung0, _read_exif_editor
from .vlm import VLMClient
from .agents import authenticity, rung1b


def _auto_reject_gate(rec: CaseRecord, cfg: EconomicConfig) -> None:
    """If the case is a REJECT and auto-reject is disabled, downgrade to escalate.

    Shopee guidance: keep auto-rejection off until the cost of wrongly rejecting a
    valid claim is known. Applies to every reject path (Rung 0 duplicate,
    Authenticity dispositive, convergence)."""
    if rec.decision == DECISION_REJECT and not cfg.enable_auto_reject:
        rec.decision = DECISION_ESCALATE
        rec.economic = {"route": "escalate", "claim_value_php": rec.claim_value_php,
                        "reason": "auto_reject_disabled: model reject downgraded to "
                                  "human review per config"}


def _apply_economics(rec: CaseRecord, prelim: str, cfg: EconomicConfig,
                     has_red_flag: bool = False) -> None:
    """Pricing layer for a SCORED case: refine the route by expected peso loss.

    Only ever moves a case between auto-approve and escalate (never rejects —
    that's the evidence layer's job, gated above). Because credibility is not a
    calibrated probability, p_invalid comes from configurable per-bucket base
    rates, not the raw score. Missing price / uncalibrated input -> human review.

    Layer-interaction rule: the economic "cheap claim -> auto-approve" downgrade
    applies ONLY to thin/UNFLAGGED uncertainty. If a fraud check raised a red flag
    (has_red_flag), the model's judgment is in conflict and the per-bucket base
    rate no longer describes this case — so it goes to a human regardless of claim
    value. A flagged case is never auto-paid just because it is cheap.
    """
    if prelim == DECISION_REJECT:
        _auto_reject_gate(rec, cfg)
        return

    # A resubmit is not a claim decision, so there is no exposure to price: the
    # proof doesn't depict the item, and expected-loss reasoning has nothing to
    # weigh. Without this guard economics would treat it as an escalate-bucket
    # case and could auto-approve it as "too cheap to review" — exactly how a
    # thumbs-down photo got approved for ₱109.
    if prelim == DECISION_RESUBMIT:
        rec.economic = {"route": "resubmit", "claim_value_php": rec.claim_value_php,
                        "reason": "irrelevant_proof: the proof does not depict what is "
                                  "being claimed; bounced for usable proof. "
                                  "No claim decision was made, so there is no "
                                  "expected loss to price."}
        return

    if prelim == DECISION_ESCALATE and has_red_flag:
        rec.economic = {"route": "escalate", "claim_value_php": rec.claim_value_php,
                        "reason": "flagged_conflict: a fraud check raised a red flag "
                                  "(<0.40); routed to human review regardless of "
                                  "claim value — the base rate does not describe a "
                                  "conflicted case"}
        return

    bucket = "approve" if prelim == DECISION_APPROVE else "escalate"
    p_invalid = cfg.bucket_p_invalid.get(bucket)
    decision = expected_loss_decision(rec.claim_value_php, p_invalid, cfg)
    rec.decision = (DECISION_APPROVE if decision.route == ROUTE_AUTO_APPROVE
                    else DECISION_ESCALATE)
    rec.economic = decision.to_dict()
    rec.economic["bucket"] = bucket


def _attach_exif_priors(rec: CaseRecord) -> None:
    """Rung 0's EXIF read, as a prior for Authenticity (images only)."""
    for ev in rec.evidence:
        if ev.path and ev.kind == "image" and ev.exif_editor is None:
            ev.exif_editor = _read_exif_editor(Path(ev.path))


def process_case(rec: CaseRecord, client: Optional[VLMClient] = None,
                 conn=None, session_id: Optional[str] = None,
                 econ: Optional[EconomicConfig] = None) -> CaseRecord:
    """Run one case end-to-end, writing decision/reason/credibility onto it.

    Also records per-stage wall times on `rec.stage_ms` (ephemeral attribute,
    not persisted) so eval scripts can report where the time goes.
    """
    t0 = time.perf_counter()
    sid = session_id or rec.session_id
    econ = econ or EconomicConfig.normal()
    stage_ms: dict[str, int] = {}
    rec.stage_ms = stage_ms

    def _mark(stage: str, since: float) -> float:
        now = time.perf_counter()
        stage_ms[stage] = int((now - since) * 1000)
        return now

    # --- Rung 0: deterministic pre-validation --------------------------------
    res = prevalidate(rec, conn=conn, session_id=sid)
    t = _mark("rung0", t0)
    if apply_rung0(rec, res):           # terminal (reject dup / resubmit unreadable)
        # Rung 0 resubmits are "cannot be judged" — economics must NOT touch them.
        # Expected-loss reasoning prices the risk of PAYING a claim; a bounce pays
        # nothing and denies nothing, so there is no exposure to trade against the
        # review cost. Only the auto-reject gate (duplicates) applies here.
        _auto_reject_gate(rec, econ)
        rec.runtime_ms = int((time.perf_counter() - t0) * 1000)
        return rec

    _attach_exif_priors(rec)

    # --- Rung 1a: Authenticity, solo, first ----------------------------------
    client = client or VLMClient()
    rec.set_signal(authenticity.run(rec, client))
    t = _mark("authenticity", t)

    breakdown = scoring.combine(rec)
    if breakdown.hard_reject:           # dispositive fraud -> reject, skip Rung 1b
        rec.credibility_score = breakdown.credibility_0_100
        rec.decision = DECISION_REJECT
        rec.reason_code = REASON_AUTHENTICITY_DISPOSITIVE
        _auto_reject_gate(rec, econ)
        rec.runtime_ms = int((time.perf_counter() - t0) * 1000)
        return rec

    # --- Rung 1b: Completeness / Tamper / Relevance / Defender (one call) ----
    for sig in rung1b.run(rec, client).values():
        rec.set_signal(sig)
    t = _mark("rung1b", t)

    # --- score (evidence layer) then refine by expected loss (pricing layer) -
    breakdown = scoring.score_case(rec)
    _apply_economics(rec, rec.decision, econ, has_red_flag=bool(breakdown.low_signals))
    if rec.reason_code is None:
        rec.reason_code = REASON_PASSED_PREVALIDATION
    rec.runtime_ms = int((time.perf_counter() - t0) * 1000)
    return rec


def process_and_store(rec: CaseRecord, client: VLMClient, conn,
                      session_id: Optional[str] = None,
                      econ: Optional[EconomicConfig] = None) -> CaseRecord:
    """process_case + persist the routed case to SQLite."""
    process_case(rec, client=client, conn=conn, session_id=session_id, econ=econ)
    db.upsert_case(conn, rec)
    return rec
