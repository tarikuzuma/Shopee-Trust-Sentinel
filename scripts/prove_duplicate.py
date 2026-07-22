"""
Phase B — proof the duplicate/reuse layer works against the LIVE database.

Phase A already stored a pHash for every real media file. Here we simulate a
fraudster reusing existing proof on a NEW claim: we take a file whose hash is
already in the DB and resubmit it under a different case_id, then show Rung 0:
  1. computes the submission's pHash,
  2. finds that hash ALREADY EXISTS in media_hashes (a different, earlier case),
  3. routes the case straight to REJECT (duplicate_proof) with zero VLM calls.

Run (after run_prevalidation.py has populated the DB):
  python scripts/prove_duplicate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel import media, db
from sentinel import prevalidation as pv
from sentinel.contract import CaseRecord, Evidence, DECISION_REJECT, REASON_DUPLICATE_PROOF

SESSION = "test"


def main() -> int:
    conn = db.connect()   # live DB
    db.init_db(conn)

    # Pick an IMAGE whose hash Phase A already stored (fast to re-decode).
    row = conn.execute(
        """
        SELECT case_id, filename, phash FROM media_hashes
        WHERE session_id = ? AND (filename LIKE '%.jpg' OR filename LIKE '%.png'
                                  OR filename LIKE '%.jpeg')
        ORDER BY id LIMIT 1
        """,
        (SESSION,),
    ).fetchone()
    if row is None:
        print("No stored image hashes found. Run scripts/run_prevalidation.py first.")
        return 1

    orig_case, filename, stored_hash = row["case_id"], row["filename"], row["phash"]
    print("STORED IN DB ALREADY (from Phase A):")
    print(f"  case_id = {orig_case}")
    print(f"  file    = {filename}")
    print(f"  pHash   = {stored_hash}")

    total_hashes = conn.execute(
        "SELECT COUNT(*) c FROM media_hashes WHERE session_id = ?", (SESSION,)
    ).fetchone()["c"]
    print(f"  (media_hashes holds {total_hashes} hashes for session '{SESSION}')")

    # Resubmit the SAME file under a brand-new case_id — the reuse scenario.
    new_case_id = f"REUSE_OF_{orig_case}"
    path = media.resolve(filename)
    if path is None:
        print(f"Media file {filename} not found on disk; cannot run proof.")
        return 1
    rec = CaseRecord(case_id=new_case_id, session_id=SESSION,
                     return_reason="Broken Products",
                     evidence=[Evidence(filename, Evidence.infer_kind(filename), str(path))])

    print(f"\nRESUBMITTING the same proof under a NEW claim: case_id = {new_case_id}")

    # record=False so the probe doesn't itself pollute the hash store.
    res = pv.prevalidate(rec, conn=conn, session_id=SESSION, record=False)
    pv.apply(rec, res)

    submitted_hash = rec.evidence[0].phash
    print(f"  Rung 0 computed this submission's pHash = {submitted_hash}")

    # Explicitly show the DB lookup that catches it.
    hit = db.find_duplicate_phash(conn, submitted_hash, SESSION,
                                  exclude_case_id=new_case_id,
                                  max_distance=pv.HASH_MAX_DISTANCE)
    print("\nDB LOOKUP  find_duplicate_phash(...):")
    print(f"  -> {hit}")

    print("\nVERDICT:")
    print(f"  decision    = {rec.decision}")
    print(f"  reason_code = {rec.reason_code}")
    print(f"  matched     = case {res.duplicate['case_id']} via {res.duplicate['filename']} "
          f"(hamming={res.duplicate['distance']})" if res.duplicate else "  matched     = (none)")

    ok = (rec.decision == DECISION_REJECT
          and rec.reason_code == REASON_DUPLICATE_PROOF
          and hit is not None and hit["case_id"] == orig_case)
    print("\n" + ("PROOF OK - reused proof was detected via the DB hash and auto-rejected, "
                  "no VLM call." if ok else "PROOF FAILED - see above."))
    conn.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
