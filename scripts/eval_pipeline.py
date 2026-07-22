"""
Instrumented evaluation runs for the full pipeline (Rung 0 -> 1a -> 1b -> route).

Subcommands:
  python scripts/eval_pipeline.py train   # all 10 labeled training cases
  python scripts/eval_pipeline.py valid   # 3 hand-constructed plausible-VALID cases
  python scripts/eval_pipeline.py test    # ~15-case unlabeled test-data subset
  python scripts/eval_pipeline.py report  # print summary tables from saved metrics

Design notes:
  - Resume-friendly: a case already in the DB with a decision is skipped, so the
    script can be re-run after a timeout or quota hiccup and it continues.
  - Rate-limit aware: sleeps between cases (free-tier RPM); if a case's signals
    show a 429/RESOURCE_EXHAUSTED, waits 60s and reprocesses that case once.
  - Instrumented: wraps VLMClient.analyze to log each call's wall time + token
    usage (split by text/image/video modality), tagged by which agent made it
    (recognized from the prompt prefix). Everything lands in data/eval_metrics.json
    for the report.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.loader import load_sheet
from sentinel import media, db, pipeline
from sentinel.contract import CaseRecord, Evidence
from sentinel.vlm import VLMClient

XLSX = r"C:\Users\gumba\Downloads\[OPS Hackathon Case] Order Details.xlsx"
METRICS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_metrics.json"
SLEEP_BETWEEN_CASES = 8.0

# Hand-constructed plausible-VALID cases: real test media whose content a human
# (me) verified clearly depicts the claimed problem. The rubric mandates these:
# training data is invalid-only, so without constructed valid cases we can never
# see whether the system avoids the "reject everything" trap.
VALID_CASES = [
    ("VALID_moldy_food", "Spilled Contents", ["237120460283155.jpg"],
     "food pouch with visible mold contamination inside"),
    ("VALID_broken_trays", "Broken Products", ["237391822290969.jpg"],
     "bulk order of meal trays, many visibly cracked/shattered"),
    ("VALID_cracked_watch", "Scratched", ["237452234258861.jpg"],
     "watch with shattered face, held with packaging debris"),
]

# Test subset: mix of image and video cases for balanced timing stats.
# Excludes the 3 files used in VALID_CASES (already evaluated there).
TEST_SUBSET = [
    "237045122294597", "237122422277782", "237135552260994", "237187828216410",
    "237198990271205", "237204022230019", "237354516249529", "237880150258407",
    "237889657242532", "237889661294852", "237708362275375", "237472035223977",
    "237634479270466", "237709757200435", "237135461268496",
]


def _load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {"cases": {}}


def _save_metrics(m: dict) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(m, indent=1), encoding="utf-8")


def _instrument(client: VLMClient, call_log: list) -> None:
    """Wrap client.analyze to record wall time + token usage per call."""
    orig = client.analyze

    def wrapped(*args, **kw):
        prompt = kw.get("prompt", args[0] if args else "")
        stage = ("authenticity" if str(prompt).startswith("You are a forensic")
                 else "rung1b" if str(prompt).startswith("You are four independent")
                 else "other")
        t0 = time.perf_counter()
        try:
            return orig(*args, **kw)
        finally:
            secs = time.perf_counter() - t0
            u = client.last_usage
            entry = {"stage": stage, "secs": round(secs, 2)}
            if u is not None:
                entry.update({
                    "prompt_tokens": u.prompt_token_count or 0,
                    "output_tokens": u.candidates_token_count or 0,
                    "thinking_tokens": getattr(u, "thoughts_token_count", 0) or 0,
                    "total_tokens": u.total_token_count or 0,
                })
                for mtc in (u.prompt_tokens_details or []):
                    entry[f"{mtc.modality.name.lower()}_tokens"] = mtc.token_count
            call_log.append(entry)

    client.analyze = wrapped


def _had_rate_limit(rec: CaseRecord) -> bool:
    return any("429" in (s.reason_string or "") or "RESOURCE_EXHAUSTED" in (s.reason_string or "")
               for s in rec.signals.values())


def _run_cases(recs: list[CaseRecord], session: str, metrics: dict) -> None:
    conn = db.connect()
    db.init_db(conn)
    client = VLMClient()

    for i, rec in enumerate(recs):
        existing = db.get_case(conn, session, rec.case_id)
        if existing is not None and existing.decision is not None and not _had_rate_limit(existing):
            print(f"[skip] {rec.case_id} already judged: {existing.decision}")
            continue

        call_log: list = []
        _instrument(client, call_log)
        try:
            media.attach_paths(rec)
            pipeline.process_and_store(rec, client, conn, session_id=session)
            if _had_rate_limit(rec):
                print(f"[429 ] {rec.case_id} rate-limited; waiting 60s and retrying once...")
                time.sleep(60)
                call_log.clear()
                pipeline.process_and_store(rec, client, conn, session_id=session)
        finally:
            client.analyze = VLMClient.analyze.__get__(client)  # unwrap

        key = f"{session}/{rec.case_id}"
        metrics["cases"][key] = {
            "session": session, "case_id": rec.case_id,
            "true_label": rec.true_label, "reason": rec.return_reason,
            "kinds": sorted({e.kind for e in rec.evidence if e.path}),
            "decision": rec.decision, "credibility": rec.credibility_score,
            "reason_code": rec.reason_code, "runtime_ms": rec.runtime_ms,
            "stage_ms": getattr(rec, "stage_ms", {}),
            "calls": call_log,
            "signals": {n: {"score": s.score, "conf": s.confidence,
                            "applicable": s.applicable, "verdict": s.verdict,
                            "reason": s.reason_string[:220]}
                        for n, s in rec.signals.items()},
        }
        _save_metrics(metrics)

        sigs = " ".join(
            f"{n[:4]}={'N/A' if not s.applicable else f'{s.score:.2f}'}"
            for n, s in rec.signals.items())
        print(f"[done] {rec.case_id}  cred={rec.credibility_score} "
              f"{rec.decision:9} ({rec.reason_code})  {sigs}  "
              f"{rec.runtime_ms}ms")

        if i < len(recs) - 1:
            time.sleep(SLEEP_BETWEEN_CASES)
    conn.close()


def cmd_train(metrics: dict) -> None:
    recs = load_sheet(XLSX, "Training Data", session_id="train_eval")
    print(f"=== TRAINING EVAL: {len(recs)} labeled cases ===")
    _run_cases(recs, "train_eval", metrics)


def cmd_valid(metrics: dict) -> None:
    recs = []
    for cid, reason, files, note in VALID_CASES:
        rec = CaseRecord(case_id=cid, session_id="valid_eval", return_reason=reason,
                         evidence=[Evidence(f, Evidence.infer_kind(f)) for f in files])
        rec.true_label = "valid (constructed)"
        recs.append(rec)
    print(f"=== CONSTRUCTED-VALID EVAL: {len(recs)} cases (must NOT reject) ===")
    _run_cases(recs, "valid_eval", metrics)


def cmd_test(metrics: dict) -> None:
    all_recs = load_sheet(XLSX, "Test Data", session_id="test_eval")
    recs = [r for r in all_recs if r.case_id in TEST_SUBSET]
    print(f"=== TEST-DATA EVAL: {len(recs)} of {len(all_recs)} cases (unlabeled) ===")
    _run_cases(recs, "test_eval", metrics)


def cmd_report(metrics: dict) -> None:
    cases = metrics["cases"]
    for session in ("train_eval", "valid_eval", "test_eval"):
        rows = [c for c in cases.values() if c["session"] == session]
        if not rows:
            continue
        print(f"\n===== {session} ({len(rows)} cases) =====")
        for c in rows:
            sigs = " ".join(
                f"{n[:4]}={'N/A' if not s['applicable'] else format(s['score'], '.2f')}"
                for n, s in c["signals"].items())
            print(f"  {c['case_id'][:20]:20} {str(c['true_label'] or '?')[:24]:24} "
                  f"cred={str(c['credibility']):>5} {str(c['decision']):9} {sigs}")

    # timing + token aggregates
    all_calls = [call for c in cases.values() for call in c.get("calls", [])]
    for stage in ("authenticity", "rung1b"):
        sc = [c for c in all_calls if c["stage"] == stage]
        if not sc:
            continue
        avg_s = sum(c["secs"] for c in sc) / len(sc)
        avg_in = sum(c.get("prompt_tokens", 0) for c in sc) / len(sc)
        avg_out = sum(c.get("output_tokens", 0) + c.get("thinking_tokens", 0) for c in sc) / len(sc)
        print(f"\n{stage}: {len(sc)} calls, avg {avg_s:.1f}s, "
              f"avg {avg_in:.0f} in / {avg_out:.0f} out tokens")


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    metrics = _load_metrics()
    {"train": cmd_train, "valid": cmd_valid, "test": cmd_test,
     "report": cmd_report}[cmd](metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
