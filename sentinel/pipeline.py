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
    CaseRecord, DECISION_REJECT, REASON_AUTHENTICITY_DISPOSITIVE,
    REASON_PASSED_PREVALIDATION,
)
from .prevalidation import prevalidate, apply as apply_rung0, _read_exif_editor
from .vlm import VLMClient
from .agents import authenticity


def _attach_exif_priors(rec: CaseRecord) -> None:
    """Rung 0's EXIF read, as a prior for Authenticity (images only)."""
    for ev in rec.evidence:
        if ev.path and ev.kind == "image" and ev.exif_editor is None:
            ev.exif_editor = _read_exif_editor(Path(ev.path))


def process_case(rec: CaseRecord, client: Optional[VLMClient] = None,
                 conn=None, session_id: Optional[str] = None) -> CaseRecord:
    """Run one case end-to-end, writing decision/reason/credibility onto it."""
    t0 = time.perf_counter()
    sid = session_id or rec.session_id

    # --- Rung 0: deterministic pre-validation --------------------------------
    res = prevalidate(rec, conn=conn, session_id=sid)
    if apply_rung0(rec, res):           # terminal (reject dup / escalate quality)
        rec.runtime_ms = int((time.perf_counter() - t0) * 1000)
        return rec

    _attach_exif_priors(rec)

    # --- Rung 1a: Authenticity, solo, first ----------------------------------
    client = client or VLMClient()
    rec.set_signal(authenticity.run(rec, client))

    breakdown = scoring.combine(rec)
    if breakdown.hard_reject:           # dispositive fraud -> reject, skip Rung 1b
        rec.credibility_score = breakdown.credibility_0_100
        rec.decision = DECISION_REJECT
        rec.reason_code = REASON_AUTHENTICITY_DISPOSITIVE
        rec.runtime_ms = int((time.perf_counter() - t0) * 1000)
        return rec

    # --- Rung 1b: the remaining checks (slot in as built) --------------------
    # rec.set_signal(completeness.run(rec, client))
    # rec.set_signal(tamper.run(rec, client))
    # rec.set_signal(relevance.run(rec, client))
    # rec.set_signal(defender.run(rec, client))

    # --- score + route on whatever signals exist -----------------------------
    scoring.score_case(rec)
    if rec.reason_code is None:
        rec.reason_code = REASON_PASSED_PREVALIDATION
    rec.runtime_ms = int((time.perf_counter() - t0) * 1000)
    return rec


def process_and_store(rec: CaseRecord, client: VLMClient, conn,
                      session_id: Optional[str] = None) -> CaseRecord:
    """process_case + persist the routed case to SQLite."""
    process_case(rec, client=client, conn=conn, session_id=session_id)
    db.upsert_case(conn, rec)
    return rec
