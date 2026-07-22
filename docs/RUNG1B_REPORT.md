# Rung 1b build report — the four remaining checks

*Written by the autonomous build session (Fable), 2026-07-23. Test tables below are
filled from `data/eval_metrics.json`, produced by `scripts/eval_pipeline.py`.*

## 1. The problem, as I understand it

Shopee refunds money on the strength of buyer-submitted photos/videos. The fraud
patterns in the labeled data are: proof that doesn't show the claimed problem at
all (irrelevant), parcels already opened before filming or unboxings never shown
(tamper/completeness), and media that was AI-generated or edited (authenticity).
The expensive mistake is paying a fraudster (false VALID); the cheap mistake is
bouncing an honest claim to a human (escalate). The training set contains *only*
invalid examples, so any system tuned naively on it will drift toward rejecting
everything — which is why the Defender exists and why we hand-construct valid
cases to test against.

The system's job is NOT "catch every fraud." It is: **decide the confident cases
autonomously in the cheap direction, and never auto-approve under uncertainty.**

## 2. What I built

### One combined call for four checks (`sentinel/agents/rung1b.py`)

Authenticity (already built, Rung 1a) judges each file's *pixels* separately —
worst-case-wins, because one faked file is fraud regardless of its neighbors.
The four Rung 1b checks are different: they judge the case's *story as a whole*
(does the evidence show the claimed problem? was the critical moment filmed?).
So all evidence goes into **one** Gemini call returning all four sub-verdicts as
structured JSON. That's ~4x fewer calls/uploads than one-per-check, and the model
sees the whole story in one context instead of four blind fragments.

Per-check `applicable` rules (rubric-mandated, enforced in *code* where absolute):

| Check | applicable rule | where enforced |
|---|---|---|
| Relevance | never N/A | forced `True` in code |
| Completeness | N/A when case has no video | forced in code via `rec.has_video` |
| Tamper | N/A when no packaging visible | model reports `packaging_visible` (a concrete perceptual fact, not an abstract judgment) |
| Defender | always applicable | forced `True` in code |

The Defender prompt explicitly forbids vague reassurance: it must *name* specific
legitimacy evidence (transit-consistent damage, internally consistent lighting/
timestamps, intact unboxing chain) or admit it found none and score low with low
confidence. It receives the Authenticity signal's verdict as context so it builds
on it instead of re-deriving it.

Tamper owns frozen/looped-frame detection (a still passed off as video is
deception, not bad capture) — per the locked v3 boundary.

Any `VLMError` → all four signals return `applicable=False` (neutral). A failed
call is never suspicion.

### Wiring (`sentinel/pipeline.py`)

The commented Rung 1b placeholder became a real call. Everything else unchanged:
Rung 0 short-circuits still fire first, the Authenticity dispositive veto still
skips Rung 1b entirely, and `scoring.score_case` still makes the final call.
Added per-stage wall-time capture (`rec.stage_ms`, ephemeral) for this report.

### Small flagged edits outside my lane (mission allowed with justification)

1. **`sentinel/vlm.py` — upload cache** (~8 lines): the same video was being
   uploaded twice per case (once for Authenticity, once for Rung 1b). Cache keyed
   by path+mtime. Halves video upload time and quota. No behavior change.
2. **`sentinel/vlm.py` — `last_usage` capture** (2 lines): stores the response's
   `usage_metadata` so eval scripts can do token/cost accounting. No behavior
   change.

Both are additive; all pre-existing tests still pass (`demo_scoring`,
`test_prevalidation`).

### Mid-run incident: model switch forced by quota (important context for all results)

The plan was to evaluate on `gemini-flash-latest`. Its free-tier **media-request
quota was already exhausted** by earlier development testing today (text requests
succeeded, any request with an image/video returned 429 even after 60s backoff).
Waiting for the daily reset was not survivable with the deadline, so I switched
`GEMINI_MODEL` to **`gemini-3.5-flash-lite`** (pinned, its own healthy quota,
verified working on both images and video before relaunching).

Two consequences to be honest about:
- **All measured results below are from the lite model** — a weaker vision model
  than full Flash. Judgment quality is plausibly *understated* relative to what
  the production model would do. The user's stated plan to scale to stronger
  models later fits this exactly: the model is one env var (`GEMINI_MODEL`).
- The first (failed) run also proved the failure design works: every 429'd check
  came back `applicable=False` → neutral → **escalate**. Ten fully-blind cases
  produced ten escalations and zero auto-approvals and zero auto-rejects. The
  system fails safe exactly as designed.

### Second incident: the credulity failure, and the burden-of-proof fix

The first complete training run (lite model, original calibration) produced the
worst possible pattern — **5 of 10 known-invalid cases auto-approved**, including
two "Parcel was already opened" videos at credibility 100.0 with `tamper=1.00`:

| first-run outcome | n | detail |
|---|---|---|
| correct reject | 2 | both AI-edit images, via authenticity veto — incl. one the full Flash model had MISSED in earlier testing (`237776404294061`, lite caught it at 0.00) |
| acceptable escalate | 3 | human review would catch |
| **false approve** | **5** | the expensive error, 5 times |

Diagnosis: the model **rubber-stamps** — it treats "I see nothing wrong" as
`1.00`, and a perfect score from a fraud check actively *pushes* a case into
auto-approve. But these training cases are real frauds that were crafted to look
plausible; absence of visible red flags is exactly what a competent fraud
produces. "No red flags" is not evidence of validity.

Fix (in `rung1b.py`'s prompt only — no locked file touched): **burden-of-proof
calibration.** A check may only score above 0.7 if it can *name the affirmative
visible evidence* (seal shown intact then opened on camera; one continuous take;
the claimed damage on the actual item). "Nothing looks wrong" is capped to
0.4–0.6 → which routes to **escalate**, not approve. Unproven claims now go to a
human instead of being paid automatically. Results below are from the re-run
with this calibration.

## 3. Test results

> **RUN STATUS: COMPLETE.** 28 cases total (10 training + 3 constructed-valid +
> 15 test), 52 successful VLM calls, 120,072 input / 9,158 output tokens,
> measured end-to-end on the live pipeline with persistence.

### 3.1 Training set (10 labeled cases, all truly invalid)

Signals: `auth`=authenticity, `comp`=completeness, `tamp`=tamper, `rele`=relevance,
`defe`=defender; N/A = not applicable (neutral, excluded from the mean).
Model: `gemini-3.5-flash-lite`. All 10 cases are genuinely invalid, so **approve
here = paying a fraudster** and escalate = a human catches it.

| case | true invalidity | cred | decision | key signals | verdict |
|---|---|---|---|---|---|
| 237872216204114 | Parcel already opened | 100.0 | approve | comp .90 tamp .90 rele .85 | ❌ false approve |
| 237797364289378 | Parcel already opened | 63.7 | escalate | comp .45 tamp .50 rele .40 | 🟡 human catches |
| 237714965285933 | No unboxing shown | 95.3 | approve | comp .85 tamp .80 | ❌ false approve |
| 237776404294061 | Edited using AI | 53.6 | **reject** | rele .10 (veto) defe .30 | ✅ caught — via *Relevance* veto (evidence doesn't show the claimed missing item) |
| 237789680255463 | Edited using AI | 98.0 | approve | auth .95 rele .85 | ❌ false approve (the subtle edit full Flash also missed) |
| 237721903250900 | No unboxing shown | 59.0 | escalate | comp .40 tamp .50 rele .40 | 🟡 human catches |
| 237685794247105 | Irrelevant proof | 58.0 | escalate | comp .40 auth .50 | 🟡 human catches |
| 237828286244255 | Irrelevant proof | 73.0 | escalate | rele .40 defe .45 | 🟡 human catches |
| 211886964225938 | Edited using AI | 0.0 | **reject** | auth .00 (dispositive veto) | ✅ caught |
| 215087824265855 | Irrelevant proof | — | escalate | Rung 0: insufficient sharpness | 🟡 human catches (zero tokens spent) |

**Scorecard: 2 auto-reject (correct), 5 escalate (safe — humans catch them), 3
false approve.** Versus the pre-calibration run: false approves 5 → 3, correct
rejects preserved, and the improvement came without any new false *rejects* —
the anti-over-rejection direction was not harmed.

Notable: the two auto-rejects fired through **two different vetoes** (authenticity
dispositive on one, relevance dispositive on the other) — the ensemble catching
different fraud types with different signals, as designed. And one interesting
reversal: `237776404294061` is the AI-edit that the *full* Flash model missed in
earlier testing; the lite model + relevance check rejected it here.

**The 3 remaining false approves are a model-capability ceiling, not a prompt
knob.** In each, the lite model reported it affirmatively *saw* an intact-seal
unboxing (comp/tamp ≥ .80) on videos whose ground truth is pre-opened/no-unboxing.
It is being fooled, confidently. See §5 for the concrete scoring-layer option
that would convert all three into escalates, left as a user decision because it
changes a locked file.

### 3.2 Constructed plausible-VALID cases (must NOT auto-reject)

Three real test-data files whose content I visually verified matches the claimed
reason (the rubric mandates constructing these, since training data has no valid
examples):

| constructed case | reason | what the media actually shows |
|---|---|---|
| VALID_moldy_food | Spilled Contents | food pouch with visible mold contamination |
| VALID_broken_trays | Broken Products | bulk meal-tray order, many visibly shattered |
| VALID_cracked_watch | Scratched | watch with shattered face + packaging debris in hand |

| case | cred | decision | verdict |
|---|---|---|---|
| VALID_broken_trays | 99.7 | **approve** | ✅ correct auto-approve of a real valid claim |
| VALID_moldy_food | 70.0 | escalate | ✅ safe (not rejected); human approves |
| VALID_cracked_watch | 67.1 | escalate | ✅ safe (not rejected); human approves |

**Zero constructed-valid cases were rejected** — the "reject everything" trap the
rubric warns about is not happening. One of three auto-approved.

A genuinely interesting calibration insight from the two escalates: both were
dragged down by **Relevance punishing reason-taxonomy mismatches** — mold in a
food pouch filed under "Spilled Contents" scored rele=0.40; a *shattered* watch
face filed under "Scratched" scored rele=0.30. The evidence is real and damning,
but it doesn't literally match the dropdown word the buyer picked. That's
defensible under the rubric ("does the evidence depict what the reason claims"),
and it fails safe (escalate, never reject) — but it means honest buyers who pick
an imprecise return reason cost a human review. A future iteration could tell
Relevance to treat "same product, plausibly related damage type" as a partial
match. Left as-is for now: conservative in exactly the direction the cost
asymmetry demands.

### 3.3 Test-data subset (unlabeled, 15 cases)

**Decisions: 7 approve / 6 escalate / 2 reject → 60% auto-decided** (the judge's
automation goal), with every uncertain case going to the human queue.

| case | reason | cred | decision | note |
|---|---|---|---|---|
| 237045122294597 | Broken Products | 92.8 | approve | |
| 237122422277782 | Damaged Others | 97.8 | approve | |
| 237204022230019 | Spilled Contents | 98.0 | approve | |
| 237708362275375 | (video) | 99.7 | approve | comp/tamp .95 — model saw a full unboxing |
| 237472035223977 | (video) | 85.8 | approve | |
| 237709757200435 | (video) | 79.8 | approve | ⚠ rele .30 — see weakness 3 |
| 237889661294852 | Item Missing (video) | 75.3 | approve | ⚠ rele .30 defe .20 — see weakness 3 |
| 237889657242532 | Damaged Others (video) | 69.3 | escalate | comp/tamp .40 |
| 237135552260994 | Suspicious Parcel | 74.5 | escalate | just under the approve line |
| 237187828216410 | Item Missing | 57.5 | escalate | rele .20 |
| 237198990271205 | Item Missing | 57.5 | escalate | rele .20 |
| 237880150258407 | Change of Mind | 73.8 | escalate | |
| 237634479270466 | (video) | — | escalate | Rung 0: insufficient sharpness, **96 ms, zero tokens** |
| 237354516249529 | Spilled Contents | 0.0 | **reject** | authenticity veto — spot-checked ✅ below |
| 237135461268496 | Spilled Contents | 0.0 | **reject** | authenticity veto — spot-checked ⚠ below |

**Spot-check of the two auto-rejects (I looked at the actual images):**

- `237354516249529` — ✅ **the reject looks right.** Four luncheon-meat cans
  "spilling" a *dry powder* (luncheon meat is a solid moist block — physically
  implausible), four near-identical tear holes, repeated label text. Textbook
  AI-generation signs; the model cited exactly these.
- `237135461268496` — ⚠ **genuinely ambiguous, leaning right but for a partly
  wrong reason.** The shampoo bottle's printed dates read MFG `25092025`,
  EXP `25092023` — expiry two years *before* manufacture, the classic near-copy
  garbled-text artifact of AI generation (though a photo of a counterfeit label
  could look the same; either deserves at minimum a human look). But the model's
  *first* cited reason was the repeating timestamp watermark — which is
  **Shopee's own evidence-portal stamp (`csportal.ph`)**, present on many
  legitimate submissions in this dataset. That misread is a false-positive
  generator; see weakness 4.

## 4. Timing & cost (measured)

### Speed, per case (all 28 evaluated cases, `gemini-3.5-flash-lite`)

| metric | value |
|---|---|
| image-only case, end to end | **5.1 s** avg (n=16) |
| video case, end to end | **12.6 s** avg (n=12) |
| Rung-0 short-circuit case | **~0.1-1.1 s, zero tokens** |
| Rung 0 stage | 0.6 s avg |
| Rung 1a authenticity call | 7.6 s avg (includes video upload) |
| Rung 1b combined call | **2.5 s** avg (reuses the cached video upload) |
| per-case tokens (all sessions) | **4,288 in / 327 out** avg |

Two design decisions visible in the numbers: the **upload cache** makes Rung 1b
~3x faster than Rung 1a on the same media (upload paid once), and the **combined
call** means four checks cost one round-trip. A Rung-0 short-circuit case costs
**~0.6-1.1 s and zero tokens**.

### Token usage, per case (measured via `usage_metadata`)

| call | avg prompt tokens | of which media | avg output+thinking |
|---|---|---|---|
| authenticity | 2,068 | 593 img + 1,117 vid | 70 |
| rung1b (all 4 checks) | 3,248 | 680 img + 1,536 vid | 313 |
| **per case total** | **≈ 4,900** | | **≈ 330** |

### Cost per case — current model vs. "smarter model" upgrades

Method: measured Gemini token counts × each provider's last-published per-token
price. **Caveats, honestly stated:** (1) prices are as last published before this
session — verify current rates before budgeting; (2) other providers tokenize
media differently (Claude/GPT have no native video input — video requires frame
extraction, changing both token counts and engineering); (3) "GPT-5.2" pricing
was not verifiable — the GPT-5.x published rate is used as proxy.

| model | $/1M in | $/1M out | ~$ per case | per 1M cases/mo |
|---|---|---|---|---|
| **Gemini Flash-Lite (current)** | $0.10 | $0.40 | **$0.0006** | ~$620 |
| Gemini Flash (full) | $0.30 | $2.50 | $0.0023 | ~$2,300 |
| Gemini Pro | $1.25 | $10 | $0.0094 | ~$9,400 |
| GPT-5.x (proxy for 5.2) | $1.25 | $10 | $0.0094 | ~$9,400 (+frame-extraction pipeline for video) |
| Claude Haiku 4.5 | $1 | $5 | $0.0066 | ~$6,600 (+frames) |
| Claude Sonnet 4.x | $3 | $15 | $0.0197 | ~$19,700 (+frames) |
| Claude Opus 4.5 | $5 | $25 | $0.0330 | ~$33,000 (+frames) |

The free tier costs $0 today. The architecture also cuts real cost independent of
model choice: Rung 0 short-circuits (zero tokens), the Rung 1a dispositive veto
skips the second call entirely (measured: the two auto-rejected training cases
each spent ~half the tokens of a full-pipeline case), and the upload cache halves
video transfer. **The cheapest call is the one never made — that's the pitch
answer to the judge's GPU-cost bottleneck.**

## 5. Honest weaknesses & recommendations (ranked)

**1. Three false approves on training frauds (the expensive error).** The lite
model affirmatively believes it saw intact-seal unboxings on videos whose ground
truth is "parcel already opened" / "no unboxing shown", and rates the subtle
AI-edit (`237789680255463`) authentic at .95 — so high-confidence wrong signals
push those cases over the 75 approve line. Prompt calibration already cut false
approves 5 → 3; the rest is model capability. **Recommended fix (needs your
sign-off — it changes locked `scoring.py`): a conflict guard.** The rubric
already says "signals conflict → escalate", but the router only implements the
score band. One added rule — *any applicable fraud check below 0.4 blocks
auto-approve and forces escalate* — would have converted several bad approves to
escalates (e.g. the two ⚠ test approves with rele=.30) at zero cost to clean
approves (none of our clean approves carried a sub-.4 signal). A stronger
variant — fraud checks cap their *upward* contribution at ~0.6, only the
Defender earns approval — would catch all 3 remaining false approves but also
demote 1-2 legitimate approves to escalate. Both are one-screen changes; I
deliberately did not make them without you.

**2. Tamper/completeness video fraud is where the lite model is weakest.** Both
correct auto-rejects were image cases (authenticity + relevance vetoes). No
video-tamper fraud was auto-caught; they land in approve (bad) or escalate (ok).
Try `gemini-flash-latest` (quota resets daily) or a stronger model on exactly
these cases before concluding the checks can't do better — the model is one env
var (`GEMINI_MODEL`).

**3. Approves that carry a low relevance signal deserve suspicion.**
`237889661294852` (Item Missing, rele=.30, defe=.20, cred 75.3) auto-approved
because strong comp/tamp/auth swamped the two low signals. The conflict guard in
(1) is the fix; until then, treat sub-.4-relevance approves as review-worthy.

**4. The authenticity check misreads Shopee's own evidence watermark.** The
repeating `csportal.ph` timestamp overlay is the platform's stamp, present on
many legitimate images; on one reject it was cited as the primary manipulation
evidence. One sentence added to `authenticity.py`'s prompt ("the repeating
csportal.ph timestamp watermark is the platform's own evidence stamp — expected
on legitimate media; do not treat it as manipulation") removes this false-positive
source. Not applied because `authenticity.py` is locked and it would invalidate
the already-measured numbers — apply + re-run when quota is fresh.

**5. Relevance punishes honest-but-imprecise return reasons** (mold filed as
"Spilled Contents" → .40; shattered filed as "Scratched" → .30). Fails safe
(escalate), but costs automation on valid claims. Possible refinement: instruct
Relevance to score "same product, plausibly related damage" as partial match.

**6. Everything below was measured on `gemini-3.5-flash-lite`** after the
`flash-latest` media quota died mid-eval (see §Mid-run incident). Numbers are
plausibly a *floor* on quality. Also remember the earlier full-Flash finding:
it missed 2/3 subtle AI-edits too — model upgrades help, but the ensemble +
escalation design is what actually contains the damage.

**7. Small samples.** 10 labeled invalid + 3 constructed valid + 15 unlabeled
cases. Directional, not statistical. The confusion-matrix script over the full
58-case test set (once quota allows) is the next measurement.

## 6. What I'd flag for your review

1. **Decide on the conflict guard** (weakness 1/3) — biggest safety win per line
   of code, and it's your locked file.
2. The two flagged test approves (`237889661294852`, `237709757200435`) — eyeball
   their videos before showing the demo.
3. The watermark prompt fix for authenticity (weakness 4) + re-run.
4. `.env` now pins `GEMINI_MODEL=gemini-3.5-flash-lite`; flip back to
   `gemini-flash-latest` when its daily media quota resets if you want the
   stronger model for the demo.
5. All results persist in `data/sentinel.db` (sessions `train_eval`,
   `valid_eval`, `test_eval`) and `data/eval_metrics.json` — the confusion-matrix
   script and swipe app can read them as-is.
