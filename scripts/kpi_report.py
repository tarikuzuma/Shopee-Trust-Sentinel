"""
KPI report — the full-pipeline scorecard: speed, cost, accuracy, and peso savings.

Pulls timing/token data from the real eval runs (data/eval_metrics.json) and
re-derives current routing (scoring + conflict guard + economics) from the cached
VLM signals so decisions reflect the CURRENT pipeline without re-spending quota.

Sections:
  A. Coverage        — how many cases, images vs videos
  B. Speed           — per case / per image / per video / per stage / per VLM call
  C. Cost            — tokens -> ₱ on the current model, vs bigger models
  D. Accuracy        — TP/FN/TN/FP on the labeled set (positive = invalid)
  E. Economic impact — labor saved vs the human-reviews-everything status quo,
                       minus expected fraud slippage = net ₱ saved, projected

Run:  python scripts/kpi_report.py            # print + write docs/KPI_REPORT.md
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sentinel import scoring, pipeline
from sentinel.contract import CaseRecord, SignalOutput
from sentinel.economics import EconomicConfig
from sentinel.loader import load_sheet

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "data" / "eval_metrics.json"
XLSX = r"C:\Users\gumba\Downloads\[OPS Hackathon Case] Order Details (1) (1).xlsx"
USD_PHP = 58.0  # [ILLUSTRATIVE] fx for the cost projection

# Constructed-valid cases reuse a real test order's media; map to its order id so
# the claim value resolves (their case_id is VALID_*, not an order id).
CONSTRUCTED_ORDER = {
    "VALID_moldy_food": "237120460283155",
    "VALID_broken_trays": "237391822290969",
    "VALID_cracked_watch": "237452234258861",
}

# per-1M-token prices (USD): (input, output). Last published rates; verify before use.
MODEL_PRICES = {
    "Gemini Flash-Lite (current)": (0.10, 0.40),
    "Gemini Flash (full)": (0.30, 2.50),
    "Gemini Pro": (1.25, 10.0),
    "GPT-5.x (proxy)": (1.25, 10.0),
    "Claude Haiku 4.5": (1.0, 5.0),
    "Claude Sonnet 4.x": (3.0, 15.0),
    "Claude Opus 4.5": (5.0, 25.0),
}


def _price_index() -> dict:
    idx = {}
    for sheet in ("Training Data", "Test Data"):
        try:
            for r in load_sheet(XLSX, sheet, "kpi"):
                if r.claim_value_php is not None:
                    idx[r.case_id] = r.claim_value_php
        except Exception:  # noqa: BLE001
            pass
    return idx


def _true_class(true_label):
    if not true_label:
        return None
    return "valid" if str(true_label).lower().startswith("valid") else "invalid"


def _reroute(case: dict, claim, cfg):
    """Rebuild a CaseRecord from cached signals + claim, return current decision."""
    rec = CaseRecord(case_id=case["case_id"], session_id=case["session"],
                     return_reason=case.get("reason") or "")
    rec.claim_value_php = claim
    for n, s in (case.get("signals") or {}).items():
        rec.set_signal(SignalOutput(n, score=s["score"], confidence=s["conf"],
                                    applicable=s["applicable"]))
    if not rec.signals:
        return case.get("decision")  # Rung 0 terminal — keep stored
    b = scoring.score_case(rec)
    pipeline._apply_economics(rec, rec.decision, cfg, has_red_flag=bool(b.low_signals))
    return rec.decision


def _avg(xs):
    return statistics.mean(xs) if xs else 0.0


def build(cfg: EconomicConfig):
    data = json.loads(METRICS.read_text(encoding="utf-8"))
    cases = list(data["cases"].values())
    price_idx = _price_index()

    # attach claim value + current decision to every case
    for c in cases:
        cid = c["case_id"]
        claim = c.get("claim_value_php")
        if claim is None:
            claim = price_idx.get(cid) or price_idx.get(CONSTRUCTED_ORDER.get(cid, ""))
        c["_claim"] = claim
        c["_decision"] = _reroute(c, claim, cfg)

    out = {"cfg": cfg}

    # A. coverage
    img = [c for c in cases if c.get("kinds") == ["image"]]
    vid = [c for c in cases if "video" in (c.get("kinds") or [])]
    out["coverage"] = {"total": len(cases), "images": len(img), "videos": len(vid)}

    # B. speed
    rt_img = [c["runtime_ms"] for c in img if c.get("runtime_ms")]
    rt_vid = [c["runtime_ms"] for c in vid if c.get("runtime_ms")]
    stage = {}
    for c in cases:
        for k, v in (c.get("stage_ms") or {}).items():
            stage.setdefault(k, []).append(v)
    calls = [call for c in cases for call in c.get("calls", []) if call.get("total_tokens")]
    call_lat = {}
    for call in calls:
        call_lat.setdefault(call["stage"], []).append(call["secs"])
    out["speed"] = {
        "img_ms": _avg(rt_img), "vid_ms": _avg(rt_vid),
        "stage_ms": {k: _avg(v) for k, v in stage.items()},
        "call_secs": {k: _avg(v) for k, v in call_lat.items()},
        "n_img_timed": len(rt_img), "n_vid_timed": len(rt_vid),
    }

    # C. cost (tokens per case)
    tok_in = [sum(call.get("prompt_tokens", 0) for call in c.get("calls", [])) for c in cases]
    tok_out = [sum(call.get("output_tokens", 0) + call.get("thinking_tokens", 0)
                   for call in c.get("calls", [])) for c in cases]
    avg_in, avg_out = _avg([t for t in tok_in if t]), _avg([t for t in tok_out if t])
    out["cost"] = {"avg_in": avg_in, "avg_out": avg_out, "models": {}}
    for name, (pin, pout) in MODEL_PRICES.items():
        usd = (avg_in / 1e6) * pin + (avg_out / 1e6) * pout
        out["cost"]["models"][name] = {"php_per_case": usd * USD_PHP,
                                       "php_per_1k": usd * USD_PHP * 1000}

    # D. accuracy on labeled sessions (positive = invalid)
    tp = fn = tn = fp = 0
    esc_inv = esc_val = 0
    for c in cases:
        tc = _true_class(c.get("true_label"))
        if tc is None:
            continue
        d = c["_decision"]
        if d == "escalate":
            if tc == "invalid":
                esc_inv += 1
            else:
                esc_val += 1
        elif tc == "invalid":
            tp += 1 if d == "reject" else 0
            fn += 1 if d == "approve" else 0
        else:
            tn += 1 if d == "approve" else 0
            fp += 1 if d == "reject" else 0
    out["accuracy"] = {"tp": tp, "fn": fn, "tn": tn, "fp": fp,
                       "esc_inv": esc_inv, "esc_val": esc_val}

    # E. economic impact vs the human-reviews-everything status quo
    # Baseline today: a human reviews EVERY case at review_cost. Our system only
    # reviews the escalations; auto-decided cases save that labor, minus the
    # expected fraud we let through on auto-approvals.
    dist = {"approve": 0, "escalate": 0, "reject": 0, None: 0}
    approve_exposure = 0.0
    p_appr = cfg.bucket_p_invalid.get("approve", 0.05)
    for c in cases:
        d = c["_decision"]
        dist[d] = dist.get(d, 0) + 1
        if d == "approve" and c.get("_claim"):
            approve_exposure += c["_claim"] * cfg.shopee_net_loss_fraction
    n = len(cases)
    auto = dist["approve"] + dist["reject"]
    labor_saved = auto * cfg.review_cost_php
    fraud_slippage = approve_exposure * p_appr
    net_saved = labor_saved - fraud_slippage
    out["economics"] = {
        "n": n, "dist": dist, "auto": auto, "escalate": dist["escalate"],
        "automation_rate": auto / n if n else 0,
        "labor_saved": labor_saved, "fraud_slippage": fraud_slippage,
        "net_saved": net_saved,
        "net_saved_per_case": net_saved / n if n else 0,
    }
    return out


def render(out: dict) -> str:
    cfg = out["cfg"]
    L = []
    cov, sp, co, ac, ec = (out["coverage"], out["speed"], out["cost"],
                           out["accuracy"], out["economics"])
    L.append("# KPI report — full pipeline\n")
    L.append(f"Model: `gemini-3.5-flash-lite`. {cov['total']} cases measured "
             f"({cov['images']} image-only, {cov['videos']} with video). Routing "
             f"re-derived from cached signals (current guard + economics). "
             f"Timing/tokens are real measurements.\n")

    L.append("## A. Speed\n")
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append(f"| image-only case, end to end | **{sp['img_ms']/1000:.1f} s** (n={sp['n_img_timed']}) |")
    L.append(f"| video case, end to end | **{sp['vid_ms']/1000:.1f} s** (n={sp['n_vid_timed']}) |")
    for k in ("rung0", "authenticity", "rung1b"):
        if k in sp["stage_ms"]:
            L.append(f"| stage: {k} | {sp['stage_ms'][k]/1000:.1f} s |")
    for k, v in sp["call_secs"].items():
        L.append(f"| VLM call: {k} | {v:.1f} s |")
    L.append("")

    L.append("## B. Cost per case\n")
    L.append(f"Measured tokens: **{co['avg_in']:,.0f} in / {co['avg_out']:,.0f} out** per case. "
             f"(fx ₱{USD_PHP}/USD, [ILLUSTRATIVE]; media tokenizes differently on non-Gemini models.)\n")
    L.append("| model | ₱/case | ₱/1,000 cases |")
    L.append("|---|---|---|")
    for name, m in co["models"].items():
        L.append(f"| {name} | ₱{m['php_per_case']:.4f} | ₱{m['php_per_1k']:,.2f} |")
    L.append("")

    L.append("## C. Accuracy (labeled set, positive = invalid proof)\n")
    total_inv = ac["tp"] + ac["fn"] + ac["esc_inv"]
    caught = ac["tp"] + ac["esc_inv"]
    L.append("| | pred reject | pred approve | escalated |")
    L.append("|---|---|---|---|")
    L.append(f"| actual INVALID | {ac['tp']} (TP) | **{ac['fn']} (FN, bad approval)** | {ac['esc_inv']} |")
    L.append(f"| actual VALID | **{ac['fp']} (FP, bad rejection)** | {ac['tn']} (TN) | {ac['esc_val']} |")
    L.append("")
    L.append(f"- Bad approvals (FN, the expensive error): **{ac['fn']}**")
    L.append(f"- Bad rejections (FP): **{ac['fp']}** — honest buyers wrongly rejected")
    if total_inv:
        L.append(f"- Fraud NOT auto-approved (caught + escalated): **{caught}/{total_inv} "
                 f"= {caught/total_inv:.0%}**")
    if (ac["tp"] + ac["fp"]):
        L.append(f"- Precision on auto-rejects: {ac['tp']}/{ac['tp']+ac['fp']} "
                 f"= {ac['tp']/(ac['tp']+ac['fp']):.0%}")
    L.append("")

    L.append("## D. Economic impact vs status quo (human reviews every case)\n")
    L.append(f"Baseline today: a human reviews all {ec['n']} cases at ₱{cfg.review_cost_php:.0f} "
             f"= ₱{ec['n']*cfg.review_cost_php:,.0f}. Our pipeline auto-decides "
             f"**{ec['auto']}/{ec['n']} = {ec['automation_rate']:.0%}**, reviewing only "
             f"the {ec['escalate']} escalations.\n")
    L.append("| line | value |")
    L.append("|---|---|")
    L.append(f"| review labor saved (auto-decided × ₱{cfg.review_cost_php:.0f}) | ₱{ec['labor_saved']:,.2f} |")
    L.append(f"| expected fraud slippage (auto-approve exposure × {cfg.bucket_p_invalid['approve']:.0%}) | −₱{ec['fraud_slippage']:,.2f} |")
    L.append(f"| **net saved on this batch** | **₱{ec['net_saved']:,.2f}** |")
    L.append(f"| net saved per case | ₱{ec['net_saved_per_case']:.2f} |")
    L.append(f"| **projected net saved / 1,000,000 cases** | **₱{ec['net_saved_per_case']*1_000_000:,.0f}** |")
    L.append("")
    L.append(f"Decision mix: {ec['dist'].get('approve',0)} approve · "
             f"{ec['dist'].get('escalate',0)} escalate · {ec['dist'].get('reject',0)} reject.")
    L.append("\n_Sample note: labeled accuracy rests on 13 cases (77% invalid, not "
             "representative); speed/cost are real per-case measurements; savings scale "
             "the measured automation rate against the ₱19 status-quo review cost._")
    return "\n".join(L)


def main() -> int:
    out = build(EconomicConfig.normal())
    report = render(out)
    print(report)
    dest = ROOT / "docs" / "KPI_REPORT.md"
    dest.write_text(report + "\n", encoding="utf-8")
    print(f"\n(wrote {dest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
