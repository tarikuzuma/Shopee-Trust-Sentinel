"""
Prints the expected-loss scenario analysis and cross-checks the dataset's
order-value statistics against the fact sheet.

Run:  python scripts/economics_report.py [path-to-xlsx]
"""
from __future__ import annotations

import sys
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Windows consoles default to cp1252; the peso sign needs UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sentinel.economics import _print_report, SCENARIOS
from sentinel.loader import load_sheet


def scenario_section() -> None:
    print("=" * 74)
    print("EXPECTED-LOSS SCENARIO ANALYSIS (per case; net benefit also per 1,000)")
    print("=" * 74)
    _print_report(n=1000)
    print("\nScenario inputs (claim value / invalid rate):")
    for name, s in SCENARIOS.items():
        print(f"  {name:8} claim ₱{s['claim_value']:,.0f}  p_invalid {s['p_invalid']:.0%}")


def dataset_section(xlsx: str) -> None:
    print("\n" + "=" * 74)
    print("DATASET ORDER-VALUE CROSS-CHECK (vs fact sheet: median ₱563 / mean ₱841)")
    print("=" * 74)
    for sheet in ("Training Data", "Test Data"):
        try:
            recs = load_sheet(xlsx, sheet, session_id="econ_report")
        except Exception as e:  # noqa: BLE001
            print(f"  {sheet}: could not load ({e})")
            continue
        vals = [r.claim_value_php for r in recs if r.claim_value_php is not None]
        missing = sum(1 for r in recs if r.claim_value_php is None)
        if not vals:
            print(f"  {sheet}: no priced cases")
            continue
        vals_sorted = sorted(vals)
        q1 = statistics.quantiles(vals, n=4)[0] if len(vals) >= 4 else vals_sorted[0]
        q3 = statistics.quantiles(vals, n=4)[2] if len(vals) >= 4 else vals_sorted[-1]
        print(f"  {sheet}: {len(recs)} cases ({missing} unpriced)")
        print(f"    median ₱{statistics.median(vals):,.0f}  mean ₱{statistics.mean(vals):,.0f}  "
              f"p25 ₱{q1:,.0f}  p75 ₱{q3:,.0f}  "
              f"min ₱{min(vals):,.0f}  max ₱{max(vals):,.0f}")


def main() -> int:
    xlsx = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\gumba\Downloads\[OPS Hackathon Case] Order Details (1) (1).xlsx"
    scenario_section()
    dataset_section(xlsx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
