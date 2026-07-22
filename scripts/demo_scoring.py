"""
Step 2 spine demo: hand-fed fake signal scores -> credibility -> decision -> DB.

No VLM yet. Proves the combiner + router + persistence work and behave correctly
on the scenarios that matter (especially the neutral-missing-signal and the
Defender-can't-rescue-fraud invariants).

Run:  python -m scripts.demo_scoring
"""
from __future__ import annotations

from sentinel.contract import CaseRecord, SignalOutput, Evidence
from sentinel.scoring import score_case, APPROVE_AT, REJECT_AT
from sentinel import db


def make(case_id, reason, sigs, evidence_kind="image") -> CaseRecord:
    rec = CaseRecord(case_id=case_id, session_id="demo", return_reason=reason,
                     evidence=[Evidence(f"{case_id}.x", evidence_kind)])
    for name, (score, conf, applicable) in sigs.items():
        rec.set_signal(SignalOutput(
            signal_name=name, score=score, confidence=conf, applicable=applicable,
            reason_string=f"{name}: score={score} conf={conf} applicable={applicable}",
        ))
    return rec


# (score, confidence, applicable) per check. Credibility: 1.0 = trustworthy.
SCENARIOS = [
    ("clean-honest-buyer", "Broken Products", {
        "authenticity": (0.95, 0.9, True), "completeness": (0.9, 0.8, True),
        "tamper": (0.9, 0.8, True), "relevance": (0.95, 0.9, True),
        "defender": (0.8, 0.7, True),
    }, "video"),

    ("photo-only-honest (completeness N/A must stay neutral)", "Broken Products", {
        "authenticity": (0.92, 0.9, True),
        "completeness": (0.0, 0.0, False),   # needs video -> inapplicable
        "tamper": (0.88, 0.8, True), "relevance": (0.9, 0.85, True),
        "defender": (0.7, 0.6, True),
    }, "image"),

    ("clear-fraud-ai-edited", "Broken Products", {
        "authenticity": (0.05, 0.95, True), "completeness": (0.4, 0.6, True),
        "tamper": (0.3, 0.7, True), "relevance": (0.2, 0.8, True),
        "defender": (0.9, 0.9, True),        # even a strong defender must NOT rescue
    }, "image"),

    ("borderline (defender should tip toward escalate/up)", "Spilled Contents", {
        "authenticity": (0.55, 0.5, True), "completeness": (0.5, 0.5, True),
        "tamper": (0.6, 0.5, True), "relevance": (0.5, 0.5, True),
        "defender": (0.9, 0.8, True),
    }, "video"),

    ("no-information (all checks inapplicable -> escalate)", "Suspicious Parcel", {
        "authenticity": (0.0, 0.0, False), "completeness": (0.0, 0.0, False),
        "tamper": (0.0, 0.0, False), "relevance": (0.0, 0.0, False),
        "defender": (0.0, 0.0, False),
    }, "image"),

    ("lone-soft-flag (1 red flag, no veto -> must escalate, NOT reject)",
     "Suspicious Parcel", {
        "relevance": (0.25, 0.6, True),      # single soft red flag, few checks apply
        "authenticity": (0.0, 0.0, False), "completeness": (0.0, 0.0, False),
        "tamper": (0.0, 0.0, False),
        "defender": (0.5, 0.4, True),
    }, "image"),

    ("converging-flags (2+ red flags -> reject)", "Suspicious Parcel", {
        "relevance": (0.25, 0.6, True), "tamper": (0.3, 0.6, True),
        "authenticity": (0.35, 0.6, True),
        "completeness": (0.0, 0.0, False),
        "defender": (0.5, 0.4, True),
    }, "video"),

    ("conflict-guard (high mean but 1 red flag -> escalate, NOT approve)",
     "Suspicious Parcel", {
        "authenticity": (0.95, 0.9, True), "completeness": (0.9, 0.85, True),
        "tamper": (0.25, 0.6, True),         # lone red flag hidden under a high mean
        "relevance": (0.9, 0.85, True),
        "defender": (0.85, 0.8, True),
    }, "video"),
]


def main():
    conn = db.connect()
    db.init_db(conn)

    print(f"thresholds:  approve >= {APPROVE_AT}   reject < {REJECT_AT}   "
          f"else escalate\n")
    print(f"{'scenario':52} {'cred':>6}  {'decision':9}  base/lift")
    print("-" * 92)

    for case_id, reason, sigs, kind in SCENARIOS:
        rec = make(case_id, reason, sigs, kind)
        b = score_case(rec)
        db.upsert_case(conn, rec)
        print(f"{case_id[:52]:52} {rec.credibility_score:6.1f}  "
              f"{rec.decision:9}  base={b.base_0_1:.2f} lift=+{b.defender_lift_0_1:.2f}"
              f"{'  [VETO:' + ','.join(b.vetoed_by) + ']' if b.hard_reject else ''}"
              f"{'  [NO INFO]' if b.no_information else ''}"
              f"{'  excl=' + ','.join(b.excluded_checks) if b.excluded_checks else ''}")

    print("\nstored session 'demo':")
    for s in db.list_sessions(conn):
        if s["session_id"] == "demo":
            print(f"  {s['n']} cases persisted to data/sentinel.db")

    # invariant assertions
    cases = {c.case_id: c for c in db.get_cases(conn, "demo")}
    photo = cases["photo-only-honest (completeness N/A must stay neutral)"]
    assert photo.decision == "approve", "missing signal must NOT sink an honest case"
    fraud = cases["clear-fraud-ai-edited"]
    # Strong-fraud veto: a highly-confident fraud signal is dispositive -> the
    # Defender is suppressed and the case auto-rejects.
    assert fraud.decision == "reject", "strong-fraud veto must auto-reject"
    noinfo = cases["no-information (all checks inapplicable -> escalate)"]
    assert noinfo.decision == "escalate", "no information must escalate"
    lone = cases["lone-soft-flag (1 red flag, no veto -> must escalate, NOT reject)"]
    assert lone.decision == "escalate", "a single soft red flag must not auto-reject"
    conv = cases["converging-flags (2+ red flags -> reject)"]
    assert conv.decision == "reject", "converging red flags must auto-reject"
    cflict = cases["conflict-guard (high mean but 1 red flag -> escalate, NOT approve)"]
    assert cflict.decision == "escalate", "a lone red flag must block auto-approval"
    print("\nOK  all scoring invariants hold.")


if __name__ == "__main__":
    main()
