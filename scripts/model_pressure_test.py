"""
Pressure-test: does a stronger Gemini model actually lift detection on the
labeled set, or is the ~50% auto-decision accuracy an architecture problem?

Runs the SAME 13 labeled cases (10 training + 3 constructed-valid) through a
candidate model, WITHOUT touching the persisted "official" eval DB rows —
results go to a separate in-memory DB and are only printed / compared here.

Run:  python scripts/model_pressure_test.py [model_name]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sentinel.loader import load_sheet
from sentinel import media, db, pipeline
from sentinel.contract import CaseRecord, Evidence
from sentinel.vlm import VLMClient

XLSX = r"C:\Users\gumba\Downloads\[OPS Hackathon Case] Order Details (1) (1).xlsx"

VALID_CASES = [
    ("VALID_moldy_food", "Spilled Contents", ["237120460283155.jpg"]),
    ("VALID_broken_trays", "Broken Products", ["237391822290969.jpg"]),
    ("VALID_cracked_watch", "Scratched", ["237452234258861.jpg"]),
]


def build_cases() -> list[CaseRecord]:
    recs = load_sheet(XLSX, "Training Data", session_id="pressure")
    for r in recs:
        media.attach_paths(r)
    for cid, reason, files in VALID_CASES:
        rec = CaseRecord(case_id=cid, session_id="pressure", return_reason=reason,
                         evidence=[Evidence(f, Evidence.infer_kind(f)) for f in files])
        rec.true_label = "valid (constructed)"
        media.attach_paths(rec)
        recs.append(rec)
    return recs


def grade(true_label: str | None, decision: str) -> str:
    if not true_label:
        return "?"
    invalid = not true_label.lower().startswith("valid")
    if decision == "escalate":
        return "escalate (safe)"
    if invalid:
        return "TP (caught)" if decision == "reject" else "FN (bad approve!)"
    return "TN (correct)" if decision == "approve" else "FP (bad reject!)"


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini-flash-latest"
    print(f"=== PRESSURE TEST: {model} on the 13-case labeled set ===\n")

    conn = db.connect(":memory:")
    db.init_db(conn)
    client = VLMClient(model=model)

    recs = build_cases()
    results = []
    t_start = time.perf_counter()
    for rec in recs:
        t0 = time.perf_counter()
        try:
            pipeline.process_and_store(rec, client, conn, session_id="pressure")
            secs = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] {rec.case_id}: {e}")
            continue
        g = grade(rec.true_label, rec.decision)
        results.append((rec.case_id, rec.true_label, rec.decision, g, secs))
        print(f"  {rec.case_id:20} label={str(rec.true_label)[:28]:28} "
              f"-> {rec.decision:9} [{g}]  ({secs:.1f}s)")
        time.sleep(5)  # be polite to the rate limit

    total = time.perf_counter() - t_start
    tp = sum(1 for r in results if r[3].startswith("TP"))
    fn = sum(1 for r in results if r[3].startswith("FN"))
    tn = sum(1 for r in results if r[3].startswith("TN"))
    fp = sum(1 for r in results if r[3].startswith("FP"))
    esc = sum(1 for r in results if r[3].startswith("escalate"))

    print(f"\n{'='*60}")
    print(f"MODEL: {model}  |  {len(results)} cases  |  {total:.0f}s total")
    print(f"  TP={tp}  FN={fn}  TN={tn}  FP={fp}  escalated={esc}")
    auto = tp + fn + tn + fp
    if auto:
        correct = tp + tn
        print(f"  Auto-decided: {auto}/{len(results)}  |  correct among auto-decided: "
              f"{correct}/{auto} = {correct/auto:.0%}")
    total_inv = tp + fn + sum(1 for r in results if r[3].startswith("escalate")
                             and not str(r[1]).lower().startswith("valid"))
    caught = tp + sum(1 for r in results if r[3].startswith("escalate")
                      and not str(r[1]).lower().startswith("valid"))
    if total_inv:
        print(f"  Fraud NOT auto-approved (caught+escalated): {caught}/{total_inv} = {caught/total_inv:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
