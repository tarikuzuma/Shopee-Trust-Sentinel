# Trust Sentinel — Validity Rubric (source of truth)

> This is our core deliverable. The dataset has no ground truth and the training
> set is invalid-only, so we don't guess labels — we define an explainable rubric
> where validity is decided by **testable questions**. The judge grades reasoning
> over raw accuracy precisely because the data can't produce clean scores.

## Scoring direction (LOCKED)

Every check outputs **credibility on 0–1 (high = trustworthy)**. They combine into
a **0–100** credibility score. High = valid, low = invalid. **Never invert.**
Every check also returns `applicable: bool` — if it can't run, it contributes
**NEUTRAL, never suspicious**.

## The four INVALID triggers (one agent each)

### 1. Relevance — "Does the evidence depict what the reason claims?"
- Reason "broken item" + evidence shows the actual damaged item → **UP**
- Evidence shows floor / empty box / wrong product / unrelated scene → **DOWN** (hard signal)
- `not_applicable`: **never** — every case has a stated reason and some evidence to compare

### 2. Completeness — "Is the critical moment actually shown?"
- Continuous unboxing, box opened on camera → **UP**
- No unboxing, or a cut at the critical moment (esp. "item missing / didn't arrive") → **DOWN**
- `not_applicable`: photo-only cases where unboxing isn't expected → **NEUTRAL** (do not penalize)

### 3. Tamper — "Was the parcel already open before filming?"
- Seal intact at start, opened on camera → **UP**
- Seal already broken / box pre-opened / contents accessible before the on-camera open → **DOWN**
- `not_applicable`: no packaging visible in evidence → **NEUTRAL**

### 4. Authenticity — "Is the media manipulated?"
- No editing artifacts; consistent compression, lighting, timestamps → **UP**
- Splices, inconsistent compression, warped regions, impossible lighting, AI artifacts → **DOWN** (hardest + highest-value)
- `not_applicable`: rare; if media unreadable/corrupt → **NEUTRAL and lean escalate**

## The DEFENDER (not-guilty vote — points OPPOSITE)

Runs alongside the four. Actively finds reasons the claim is legitimate and pulls
credibility **UP**:
- Damage consistent with normal transit
- Lighting / timestamps / metadata internally consistent
- Nothing staged; unboxing chain intact
- Only soft / single red flags rather than multiple hard ones

**Mandatory** because we've never seen a labeled VALID example — without it the
system drifts into "reject everything." Give it **real weight**.

## Decision rule (cost asymmetry baked in)

- credibility > ~75 → **AUTO-APPROVE** (valid)
- credibility < ~35 → **AUTO-REJECT** (invalid)
- 35–75, OR signals conflict, OR evidence too thin → **ESCALATE** to human

**Core principle:** *Insufficient evidence to prove invalid is NOT the same as
proven valid.* A wrong "valid" hands money to fraudsters — ambiguity escalates,
never approves.

## The trap to avoid (whole team)

Because every studied example was invalid, the natural failure is a machine that
labels everything invalid. Two guards:
1. The **Defender must carry real weight** in the formula.
2. A **single soft red flag must not sink a case** — it takes either **one hard
   trigger** or **multiple converging signals** to push below the reject line.

Tune thresholds against a small hand-built eval set that deliberately includes
plausible VALID cases we construct ourselves.
