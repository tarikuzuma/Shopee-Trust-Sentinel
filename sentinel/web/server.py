"""
Shopee Trust Sentinel — web backend.

One FastAPI process serves the single-file frontend AND a small JSON API over the
SQLite results. No build step, no Node — `uvicorn` and it's live, safe for a demo.

Endpoints:
  GET /                         -> the app shell (index.html)
  GET /api/sessions             -> [{session_id, n, started}]
  GET /api/summary?session=...  -> KPI cards + decision mix + confusion matrix
  GET /api/cases?session=...    -> case rows (filterable by decision)
  GET /api/case/{session}/{cid} -> one case's full detail (signals, economic, evidence)
  GET /api/queue?session=...    -> escalation queue (lowest credibility first)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

from .. import db
from ..contract import CaseRecord
from ..economics import EconomicConfig

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Shopee Trust Sentinel")

# Default session shown by the dashboard (the full economics run).
DEFAULT_SESSION = "test_eval"


def _true_class(true_label: Optional[str]) -> Optional[str]:
    if not true_label:
        return None
    return "valid" if str(true_label).lower().startswith("valid") else "invalid"


def _case_brief(rec: CaseRecord) -> str:
    """Deterministic plain-language brief assembled from existing fields (no LLM)."""
    if rec.economic and rec.economic.get("reason"):
        econ = rec.economic["reason"].split(":", 1)
        econ_txt = econ[1].strip() if len(econ) > 1 else rec.economic["reason"]
    else:
        econ_txt = ""
    # the most alarming applicable signal
    flags = [(n, s) for n, s in rec.signals.items()
             if n != "defender" and s.applicable and s.score < 0.5]
    flags.sort(key=lambda x: x[1].score)
    parts = []
    if rec.reason_code == "duplicate_proof":
        parts.append("Reused proof image detected (matches an earlier case).")
    elif rec.reason_code == "authenticity_dispositive_fraud":
        parts.append("Media appears AI-generated or edited.")
    elif rec.reason_code == "insufficient_evidence":
        parts.append("Proof too blurry/dark/short to judge.")
    if flags:
        n, s = flags[0]
        parts.append(f"Weakest signal: {n} ({s.score:.2f}) — {s.reason_string}")
    if econ_txt:
        parts.append(econ_txt)
    return " ".join(parts) or "No strong signals either way."


def _row(rec: CaseRecord) -> dict:
    return {
        "case_id": rec.case_id,
        "return_reason": rec.return_reason,
        "credibility": rec.credibility_score,
        "decision": rec.decision,
        "reason_code": rec.reason_code,
        "claim_value_php": rec.claim_value_php,
        "true_label": rec.true_label,
        "runtime_ms": rec.runtime_ms,
        "n_evidence": len(rec.evidence),
        "has_video": rec.has_video,
    }


@app.get("/api/sessions")
def sessions():
    conn = db.connect()
    return db.list_sessions(conn)


@app.get("/api/summary")
def summary(session: str = Query(DEFAULT_SESSION)):
    conn = db.connect()
    cases = db.get_cases(conn, session)
    n = len(cases)
    mix = {"approve": 0, "escalate": 0, "reject": 0}
    total_runtime = 0
    total_claim = 0.0
    labeled = {"tp": 0, "fn": 0, "tn": 0, "fp": 0, "esc_inv": 0, "esc_val": 0}
    for rec in cases:
        if rec.decision in mix:
            mix[rec.decision] += 1
        total_runtime += rec.runtime_ms or 0
        total_claim += rec.claim_value_php or 0
        tc = _true_class(rec.true_label)
        if tc:
            d = rec.decision
            if d == "escalate":
                labeled["esc_inv" if tc == "invalid" else "esc_val"] += 1
            elif tc == "invalid":
                labeled["tp" if d == "reject" else "fn"] += 1
            else:
                labeled["tn" if d == "approve" else "fp"] += 1

    cfg = EconomicConfig.normal()
    auto = mix["approve"] + mix["reject"]
    labor_saved = auto * cfg.review_cost_php
    approve_exposure = sum((r.claim_value_php or 0) * cfg.shopee_net_loss_fraction
                           for r in cases if r.decision == "approve")
    slippage = approve_exposure * cfg.bucket_p_invalid["approve"]
    has_labels = any(_true_class(r.true_label) for r in cases)

    return {
        "session": session, "n": n, "mix": mix,
        "automation_rate": (auto / n) if n else 0,
        "avg_runtime_ms": (total_runtime / n) if n else 0,
        "total_claim_php": total_claim,
        "labor_saved_php": labor_saved,
        "slippage_php": slippage,
        "net_saved_php": labor_saved - slippage,
        "net_saved_per_1m": ((labor_saved - slippage) / n * 1_000_000) if n else 0,
        "confusion": labeled if has_labels else None,
        "review_cost_php": cfg.review_cost_php,
    }


@app.get("/api/cases")
def cases(session: str = Query(DEFAULT_SESSION), decision: Optional[str] = None):
    conn = db.connect()
    recs = db.get_cases(conn, session, decision)
    return [_row(r) for r in recs]


@app.get("/api/queue")
def queue(session: str = Query(DEFAULT_SESSION)):
    conn = db.connect()
    return [_row(r) for r in db.get_escalated(conn, session)]


@app.get("/api/case/{session}/{case_id}")
def case_detail(session: str, case_id: str):
    conn = db.connect()
    rec = db.get_case(conn, session, case_id)
    if rec is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {
        **_row(rec),
        "brief": _case_brief(rec),
        "items": [{"item_id": i.item_id, "title": i.title, "price_php": i.price_php,
                   "listing_link": i.listing_link} for i in rec.items],
        "evidence": [{"filename": e.filename, "kind": e.kind} for e in rec.evidence],
        "signals": {n: {"score": s.score, "confidence": s.confidence,
                        "verdict": s.verdict, "applicable": s.applicable,
                        "reason": s.reason_string} for n, s in rec.signals.items()},
        "economic": rec.economic,
        "prevalidation": rec.prevalidation,
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/{path:path}")
def static_or_index(path: str):
    f = STATIC / path
    if f.is_file():
        return FileResponse(f)
    return FileResponse(STATIC / "index.html")
