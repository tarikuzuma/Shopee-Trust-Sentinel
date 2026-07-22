"""
Locks the expected-loss engine's arithmetic and safe-handling behavior.

Run:  python scripts/test_economics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.economics import (
    EconomicConfig, expected_loss_decision, scenario_metrics, scenario_report,
    ROUTE_AUTO_APPROVE, ROUTE_ESCALATE,
)

_fail = 0


def check(label, got, want, tol=0.01):
    global _fail
    ok = (got == want) if isinstance(want, (str, bool, type(None))) else abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got={got!r} want={want!r}")
    if not ok:
        _fail += 1


# --- Scenario arithmetic (the 'make no mistakes' guard) ----------------------
EXPECTED = {
    "bull":   {"net_loss_exposure": 1069.0, "incorrect_approval_loss_avoided": 101.555,
               "human_review_cost": 10.0, "wrongful_rejection_cost": 0.45,
               "delay_cost": 0.50, "net_economic_benefit": 90.605,
               "benefit_cost_ratio": 9.274, "break_even_p_invalid_full": 0.010826,
               "break_even_p_invalid_simple": 0.009354},
    "normal": {"net_loss_exposure": 420.5, "incorrect_approval_loss_avoided": 30.276,
               "human_review_cost": 19.0, "wrongful_rejection_cost": 1.84,
               "delay_cost": 1.0, "net_economic_benefit": 8.436,
               "benefit_cost_ratio": 1.386, "break_even_p_invalid_full": 0.057828,
               "break_even_p_invalid_simple": 0.045184},
    "bear":   {"net_loss_exposure": 50.0, "incorrect_approval_loss_avoided": 1.875,
               "human_review_cost": 40.0, "wrongful_rejection_cost": 9.50,
               "delay_cost": 3.0, "net_economic_benefit": -50.625,
               "benefit_cost_ratio": 0.0357, "break_even_p_invalid_full": 1.11579,
               "break_even_p_invalid_simple": 0.80},
}

print("Scenario arithmetic:")
rep = scenario_report(n=1000)
for name, exp in EXPECTED.items():
    for key, want in exp.items():
        check(f"{name}.{key}", rep[name][key], want, tol=0.02)

print("Per-1000 net benefit:")
check("bull.per1000", rep["bull"]["per_n"]["net_benefit"], 90605.0, tol=1)
check("normal.per1000", rep["normal"]["per_n"]["net_benefit"], 8436.0, tol=1)
check("bear.per1000", rep["bear"]["per_n"]["net_benefit"], -50625.0, tol=1)


# --- Decision routing --------------------------------------------------------
print("Decision routing:")
# Normal config, high-value + high p_invalid -> escalate (worth reviewing).
d = expected_loss_decision(5000.0, 0.30, EconomicConfig.normal())
check("high value+risk -> escalate", d.route, ROUTE_ESCALATE)
# Normal config, tiny-value uncertain case -> auto-approve (not worth ₱19 review).
d = expected_loss_decision(30.0, 0.30, EconomicConfig.normal())
check("low value uncertain -> auto_approve", d.route, ROUTE_AUTO_APPROVE)


# --- Value sensitivity: same p_invalid, monotone in claim value --------------
print("Value sensitivity (same p_invalid=0.08, normal cfg):")
cfg = EconomicConfig.normal()
routes = {v: expected_loss_decision(v, 0.08, cfg).route for v in (20, 100, 500, 2000)}
check("₱20 -> auto_approve", routes[20], ROUTE_AUTO_APPROVE)
check("₱2000 -> escalate", routes[2000], ROUTE_ESCALATE)
# There must be a single crossover: once it escalates it never flips back cheaper.
seq = [routes[v] for v in (20, 100, 500, 2000)]
mono = all(not (seq[i] == ROUTE_ESCALATE and seq[i + 1] == ROUTE_AUTO_APPROVE)
           for i in range(len(seq) - 1))
check("monotone approve->escalate as value rises", mono, True)


# --- Boundary at the break-even probability ----------------------------------
print("Boundary at break-even:")
cfg = EconomicConfig.normal()
d_mid = expected_loss_decision(841.0, 0.50, cfg)
p_star = d_mid.p_threshold_full
# Exactly at threshold -> escalate (tie favors human); just below -> auto-approve.
check("at p* -> escalate", expected_loss_decision(841.0, p_star, cfg).route, ROUTE_ESCALATE)
check("just below p* -> auto_approve",
      expected_loss_decision(841.0, p_star - 0.005, cfg).route, ROUTE_AUTO_APPROVE)
check("just above p* -> escalate",
      expected_loss_decision(841.0, p_star + 0.005, cfg).route, ROUTE_ESCALATE)


# --- Safe handling of bad / missing inputs -----------------------------------
print("Safe handling:")
check("missing price -> escalate",
      expected_loss_decision(None, 0.30, EconomicConfig.normal()).route, ROUTE_ESCALATE)
check("negative price -> escalate",
      expected_loss_decision(-5.0, 0.30, EconomicConfig.normal()).route, ROUTE_ESCALATE)
check("p_invalid None -> escalate",
      expected_loss_decision(500.0, None, EconomicConfig.normal()).route, ROUTE_ESCALATE)
check("p_invalid >1 -> escalate",
      expected_loss_decision(500.0, 1.4, EconomicConfig.normal()).route, ROUTE_ESCALATE)
check("p_invalid <0 -> escalate",
      expected_loss_decision(500.0, -0.1, EconomicConfig.normal()).route, ROUTE_ESCALATE)
d0 = expected_loss_decision(500.0, 0.30, EconomicConfig(shopee_net_loss_fraction=0.0))
check("zero exposure -> auto_approve", d0.route, ROUTE_AUTO_APPROVE)
check("zero exposure reason", "zero_exposure" in d0.reason, True)

# audit fields populated on a normal decision
d = expected_loss_decision(841.0, 0.30, EconomicConfig.normal())
check("audit has expected losses",
      d.expected_loss_auto_approve is not None and d.expected_loss_escalate is not None, True)

print()
if _fail:
    print(f"{_fail} check(s) FAILED")
    raise SystemExit(1)
print("All economics checks passed.")
