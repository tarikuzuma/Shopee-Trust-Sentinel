# KPI report — full pipeline

Model: `gemini-3.5-flash-lite`. 71 cases measured (37 image-only, 34 with video). Routing re-derived from cached signals (current guard + economics). Timing/tokens are real measurements.

## A. Speed

| metric | value |
|---|---|
| image-only case, end to end | **5.0 s** (n=37) |
| video case, end to end | **17.5 s** (n=34) |
| stage: rung0 | 0.6 s |
| stage: authenticity | 8.4 s |
| stage: rung1b | 2.7 s |
| VLM call: authenticity | 7.9 s |
| VLM call: rung1b | 2.7 s |

## B. Cost per case

Measured tokens: **5,982 in / 369 out** per case. (fx ₱58.0/USD, [ILLUSTRATIVE]; media tokenizes differently on non-Gemini models.)

| model | ₱/case | ₱/1,000 cases |
|---|---|---|
| Gemini Flash-Lite (current) | ₱0.0433 | ₱43.26 |
| Gemini Flash (full) | ₱0.1576 | ₱157.60 |
| Gemini Pro | ₱0.6477 | ₱647.74 |
| GPT-5.x (proxy) | ₱0.6477 | ₱647.74 |
| Claude Haiku 4.5 | ₱0.4540 | ₱453.97 |
| Claude Sonnet 4.x | ₱1.3619 | ₱1,361.90 |
| Claude Opus 4.5 | ₱2.2698 | ₱2,269.83 |

## C. Accuracy (labeled set, positive = invalid proof)

| | pred reject | pred approve | escalated |
|---|---|---|---|
| actual INVALID | 2 (TP) | **3 (FN, bad approval)** | 5 |
| actual VALID | **0 (FP, bad rejection)** | 1 (TN) | 2 |

- Bad approvals (FN, the expensive error): **3**
- Bad rejections (FP): **0** — honest buyers wrongly rejected
- Fraud NOT auto-approved (caught + escalated): **7/10 = 70%**
- Precision on auto-rejects: 2/2 = 100%

## D. Economic impact vs status quo (human reviews every case)

Baseline today: a human reviews all 71 cases at ₱19 = ₱1,349. Our pipeline auto-decides **40/71 = 56%**, reviewing only the 31 escalations.

| line | value |
|---|---|
| review labor saved (auto-decided × ₱19) | ₱760.00 |
| expected fraud slippage (auto-approve exposure × 5%) | −₱289.62 |
| **net saved on this batch** | **₱470.38** |
| net saved per case | ₱6.62 |
| **projected net saved / 1,000,000 cases** | **₱6,625,000** |

Decision mix: 31 approve · 31 escalate · 9 reject.

_Sample note: labeled accuracy rests on 13 cases (77% invalid, not representative); speed/cost are real per-case measurements; savings scale the measured automation rate against the ₱19 status-quo review cost._
