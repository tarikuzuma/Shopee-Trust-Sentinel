# Shopee Trust Sentinel

Autonomous return/refund proof review for the TRAILBLAZE Shopee AI Hackathon 2026
(Fraud Detection track). Given a return case — a stated reason plus photo/video
evidence — the system decides **approve / escalate / reject / resubmit** on its
own, using only front-of-camera signals (no IP, device, address, or order-history
data). A human only ever sees the cases the system can't safely settle itself.

Full design rationale and every locked decision: [`contex.txt`](contex.txt).
Economics derivation: [`docs/ECONOMICS.md`](docs/ECONOMICS.md). Combined-agent
prompt design: [`docs/RUNG1B_REPORT.md`](docs/RUNG1B_REPORT.md).

---

## What it does

1. Buyer submits a return case: return reason + image/video proof, against a
   known order (item title, price).
2. The pipeline below routes the case to one of four outcomes:
   - **approve** — auto-close, no reviewer touches it.
   - **reject** — auto-deny (dispositive fraud found), no reviewer touches it.
   - **resubmit** — bounced back to the buyer (evidence unreadable, or doesn't
     depict the claim); nobody reviews it, nothing is decided yet.
   - **escalate** — sent to a human, with a plain-language brief, the signal
     breakdown, and the economic justification for why a review was worth
     paying for.
3. Every case, at every exit point, gets a full row in SQLite — including the
   ones Rung 0 or Rung 1a short-circuited — so nothing disappears from the
   confusion matrix just because it was cheap to decide.
4. A FastAPI + single-file HTML dashboard (`sentinel/web/`) serves the batch:
   KPIs, a review queue, per-case drill-down with evidence playback, and a
   like-for-like comparison against a naive "escalate anything uncertain"
   baseline (see **Confidence & known limits** below for what that comparison
   does and doesn't prove).

## Requirements

- Python 3.11+ (developed against 3.13)
- A Gemini API key (`GEMINI_API_KEY`) for the two VLM calls — no other model
  provider is wired in
- No GPU, no fine-tuning, no training pipeline — this is inference-only over a
  pretrained VLM plus deterministic pre-filtering

```
fastapi, uvicorn         # web/API layer
google-genai             # Gemini client (Authenticity + Rung 1b calls)
python-dotenv            # loads GEMINI_API_KEY from .env
numpy                    # signal/threshold math
opencv-python-headless   # Rung 0: blur/brightness/frame sampling
imagehash, Pillow        # Rung 0: perceptual hash for duplicate detection
```

There is no committed `requirements.txt` / lockfile yet — the above is the
complete third-party surface (confirmed by grep against every import in
`sentinel/`), not carried over from a manifest. Pin these before treating the
build as reproducible outside this machine.

### Running it

```bash
# .env: GEMINI_API_KEY=...   (optional GEMINI_MODEL=, default gemini-flash-latest)
python -m uvicorn sentinel.web.server:app --port 8000
```

Batch scripts live in `scripts/` — `eval_pipeline.py` runs the full pipeline
over a session, `confusion_matrix.py` / `kpi_report.py` / `economics_report.py`
regenerate the graded deliverables from what's already in SQLite.

---

## Software architecture

Everything is a **rung**: each layer decides *only* the question it's
authoritative on, and hands the case downward only if it doesn't already have
an answer. Layers never re-derive something a cheaper layer upstream already
settled (e.g. Authenticity is handed the EXIF editing-software tag as a prior
instead of re-discovering it from pixels).

```
Return case (reason + photo/video + order: item title, price)
      │
      ▼
┌───────────────────────── Rung 0 — deterministic, zero tokens ─────────────────────────┐
│  sentinel/prevalidation.py   (cv2 + Pillow + imagehash, no network calls)              │
│                                                                                          │
│  • quality gate   : corrupt / too blurry / too dark / too short → RESUBMIT             │
│  • duplicate check: perceptual hash vs every prior submitted proof → REJECT if matched  │
│  • EXIF read      : editing-software tag → passed forward as a prior, never a verdict   │
└───────────────────────────────────────────────────────────────────────────────────────┘
      │ pass-through cases only, + EXIF prior attached
      ▼
┌───────────────────── Rung 1a — Authenticity (solo, runs FIRST) ───────────────────────┐
│  sentinel/agents/authenticity.py — ONE Gemini call                                     │
│  AI-generated / edited / recaptured? worst-case across evidence files.                │
│  score ≤0.15 @ confidence ≥0.85 (dispositive)  →  REJECT, STOP (skip Rung 1b entirely) │
└───────────────────────────────────────────────────────────────────────────────────────┘
      │ only non-dispositive cases reach here (fail-fast — save the token spend)
      ▼
┌───────────────────── Rung 1b — Completeness + Tamper + Relevance + Defender ───────────┐
│  sentinel/agents/rung1b.py — ONE combined Gemini call, 4 structured signals out        │
│  Relevance also knows the ordered item's TITLE, not just the return-reason string,     │
│  so a mismatch ("ordered shampoo, evidence shows grey fabric") is a checkable fact.    │
└───────────────────────────────────────────────────────────────────────────────────────┘
      │ { score, reason_string, applicable } per signal
      ▼
┌───────────────────────── Scoring combiner + decision router ──────────────────────────┐
│  sentinel/scoring.py                                                                   │
│  credibility 0–100 (higher = more trustworthy). Router gates, in order:                │
│    veto (≤0.15 @ ≥0.85)        → REJECT                                                │
│    ≥3 converging red flags     → REJECT  (outranks a rescuing mean)                    │
│    credibility > ~75           → approve bucket                                        │
│    credibility < ~35           → reject bucket (auto-reject: configurable, default ON) │
│    35–75                       → escalate bucket                                       │
│    …then, escalate-only: relevance ≤0.40 → RESUBMIT (nothing for a human to judge)     │
└───────────────────────────────────────────────────────────────────────────────────────┘
      │ approve / escalate bucket only (reject & resubmit are already final)
      ▼
┌───────────────────── Rung 2 — expected-loss / pricing layer ──────────────────────────┐
│  sentinel/economics.py, wired in sentinel/pipeline.py                                 │
│  Refines approve ↔ escalate by MINIMIZING EXPECTED PESO LOSS, not accuracy.            │
│  Escalation costs ~₱19; only worth paying when prevented fraud loss exceeds it.        │
│  p_invalid comes from configurable per-bucket base rates, NEVER raw credibility        │
│  (the score is not a calibrated probability). Missing price / bad probability          │
│  → escalate, never a silent approve. Never touches reject or resubmit.                │
└───────────────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
Persisted to SQLite (sentinel/db.py) — full audit row per case, including every
signal, the EXIF prior, the economic trace, and (Rung-0/1a short-circuits included)
the reason it stopped where it did.
      │
      ▼
sentinel/web/  — FastAPI JSON API + single-file dashboard (KPIs, decision-value
charts, review queue, per-case drawer with evidence + rerun/override controls)
```

### Why it's layered this way

- **Cheapest, most certain checks run first.** Rung 0 is free and
  deterministic; it only ever escalates a capture problem (never rejects one —
  a blurry photo from a cheap phone is not fraud) or rejects a proven
  duplicate (a signal no VLM produces as reliably).
- **The most expensive/decisive VLM check runs alone, before the other three.**
  Authenticity can settle obvious fraud by itself; only cases it doesn't
  settle pay for the remaining three-signal call.
- **Direction is never overloaded.** Every signal is credibility (high =
  trustworthy); a check that can't run returns `applicable: false` and
  contributes nothing, never suspicion.
- **"Is the proof authentic" is separated from "is reviewing this worth the
  money."** Rungs 0/1a/1b answer the first question; Rung 2 answers the
  second, using the *same* verdict under different economic assumptions — it
  never re-judges the evidence and never rejects.
- **No layer treats an unconfirmed number as a probability.** Rung 2's
  `p_invalid` is a configurable placeholder, explicitly not the model's raw
  score, because the score was never calibrated to be one.

---

## Confidence & known limits

This section exists because the deliverable is graded on reasoning and
honesty about the numbers, not just the classifier. Read it before citing any
figure from the dashboard or the KPI report as more certain than it is.

**Labeled sample is small.** `docs/KPI_REPORT.md`'s accuracy table is drawn
from **13 labeled cases** (measured: 2 TP, 3 FN, 0 FP, 1 TN, 5 escalated,
2 unreadable) — 77% invalid, not representative of a real mixed-traffic batch.
Read it for *error type* (are false approvals happening, are false rejections
happening), not as a stable rate. The training set is invalid-only by design
(per the hackathon spec), which is exactly why the test-set confusion matrix,
thin as it is, is the only real signal on false-rejection behavior.

**Every routing threshold is an untuned placeholder**, not a value derived
from this dataset: the 35/75 credibility split, Rung 0's blur/brightness/
duration floors, the duplicate-hash hamming distance, the authenticity veto
bar (≤0.15 @ ≥0.85), `ITEM_MISMATCH_MAX`, and `CONSENSUS_REJECT_MIN = 3`.
Guess wrong on any of these and the system starts rejecting or escalating a
large share of honest low-end-device photos. They need calibration against a
larger labeled set before this is production-grade, not just demo-grade.

**The economics layer's inputs are explicitly [SHOPEE INPUT REQUIRED]
placeholders**, tagged as such in `sentinel/economics.py` and
`docs/ECONOMICS.md`: the ~₱19 review cost, the per-bucket invalid base rates
(`approve≈0.05`, `escalate≈0.30`), human detection/false-reject rates, and the
cost of a wrongful rejection. The bull/normal/bear scenario spread exists
*because* these are assumptions, not facts — the "Bear" scenario deliberately
produces a **negative** net benefit (reviewing costs more than the fraud it
prevents), and that result is kept rather than hidden, because a pricing
layer that can't say "don't review this" isn't honest.

**The dashboard's "value-aware routing vs. uncertainty gate" comparison is a
same-data counterfactual, not a held-out A/B test.** Both policies are
re-costed over the *same* already-decided cases using identical `p_invalid`
and human-error assumptions, so the ₱ delta it reports isolates the effect of
routing logic alone — it is not evidence that the counterfactual policy would
perform identically if it had actually been run (routing changes which cases
a human sees, which the human's own behavior could then respond to).

**Two named model-capability ceilings, not missing checks:** the combined
Rung 1b call can be fooled by a pre-opened parcel staged to look like a clean
unboxing on video (`docs/RUNG1B_REPORT.md`); and Authenticity's fraud-veto
threshold means genuinely camera-real footage of staged fraud (real camera,
real pre-opened box) can score authenticity high — this is *by construction*,
which is why the consensus-reject gate (3 converging red flags) exists to
catch it independent of Authenticity's opinion.

**Out of scope by design, not oversight:** anything requiring backend data
(IP, device fingerprint, address, order/account history), fine-tuning (no
time, no dataset for it — API-only VLM calls), and the not-yet-built synthesis
agent / swipe app named in `contex.txt`'s build sequence.
