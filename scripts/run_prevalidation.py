"""
Phase A — timed Rung 0 run over the real datasets, persisted to the live DB.

Does NOT reset the database (schema uses CREATE TABLE IF NOT EXISTS; cases and
media hashes upsert idempotently). Reports, in order:
  1. Corruption first — any file that would not decode.
  2. Per-file speed — average ms for a VIDEO vs an IMAGE (the judge's bottleneck
     is speed/cost, so this is a headline number).
  3. Route breakdown per session.

Usage:
  python scripts/run_prevalidation.py "C:\\path\\to\\Order Details.xlsx"
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.loader import load_sheet
from sentinel import media, db
from sentinel import prevalidation as pv


def run_session(conn, xlsx: str, sheet: str, session_id: str) -> dict:
    recs = load_sheet(xlsx, sheet, session_id=session_id)
    for r in recs:
        media.attach_paths(r)

    img_ms, vid_ms = [], []
    corrupted, routes = [], {"pass": 0, "escalate": 0, "reject": 0}

    t_wall = time.perf_counter()
    for r in recs:
        res = pv.prevalidate(r, conn=conn, session_id=session_id)
        terminal = pv.apply(r, res)
        db.upsert_case(conn, r)   # persist the routed case + trace

        for m in res.evidence_qc:
            if m.get("status") == "absent":
                continue
            if m.get("status") == "corrupted":
                corrupted.append((r.case_id, m.get("filename"), m.get("note", "")))
            ems = m.get("eval_ms")
            if ems is not None:
                (vid_ms if m.get("kind") == "video" else img_ms).append(ems)

        if not terminal:
            routes["pass"] += 1
        elif r.decision == "reject":
            routes["reject"] += 1
        else:
            routes["escalate"] += 1
    wall = time.perf_counter() - t_wall

    return {
        "session": session_id, "sheet": sheet, "cases": len(recs),
        "img_ms": img_ms, "vid_ms": vid_ms, "corrupted": corrupted,
        "routes": routes, "wall_s": wall,
    }


def _avg(xs): return sum(xs) / len(xs) if xs else 0.0


def main() -> int:
    xlsx = sys.argv[1] if len(sys.argv) > 1 else None
    if not xlsx:
        print('usage: python scripts/run_prevalidation.py "<xlsx path>"')
        return 1

    conn = db.connect()      # live DB at data/sentinel.db — not reset
    db.init_db(conn)

    results = [
        run_session(conn, xlsx, "Training Data", "train"),
        run_session(conn, xlsx, "Test Data", "test"),
    ]

    all_img = [m for r in results for m in r["img_ms"]]
    all_vid = [m for r in results for m in r["vid_ms"]]
    all_corrupt = [c for r in results for c in r["corrupted"]]

    # 1) CORRUPTION FIRST
    print("=" * 70)
    print("1. CORRUPTION CHECK (files that would not decode)")
    print("=" * 70)
    if all_corrupt:
        for cid, fn, note in all_corrupt:
            print(f"  CORRUPT  case={cid}  file={fn}  ({note})")
    else:
        print("  none - every referenced media file decoded cleanly.")

    # 2) SPEED
    print("\n" + "=" * 70)
    print("2. SPEED  (Rung 0 per file, deterministic, zero API cost)")
    print("=" * 70)
    print(f"  IMAGES  n={len(all_img):3}  avg={_avg(all_img):8.1f} ms  "
          f"min={min(all_img) if all_img else 0:.1f}  max={max(all_img) if all_img else 0:.1f}")
    print(f"  VIDEOS  n={len(all_vid):3}  avg={_avg(all_vid):8.1f} ms  "
          f"min={min(all_vid) if all_vid else 0:.1f}  max={max(all_vid) if all_vid else 0:.1f}")
    tot_files = len(all_img) + len(all_vid)
    tot_wall = sum(r["wall_s"] for r in results)
    print(f"  TOTAL   {tot_files} files in {tot_wall:.2f}s wall "
          f"({tot_wall / tot_files * 1000:.1f} ms/file incl. hashing + DB writes)")

    # 3) ROUTES
    print("\n" + "=" * 70)
    print("3. ROUTES PER SESSION  (persisted to data/sentinel.db)")
    print("=" * 70)
    for r in results:
        print(f"  {r['session']:6} ({r['cases']} cases): {r['routes']}   wall={r['wall_s']:.2f}s")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
