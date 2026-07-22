"""
Rung 1b — the remaining four checks, in ONE combined VLM call.

Completeness, Tamper, Relevance, and the Defender all judge the CASE'S STORY AS A
WHOLE (unlike Authenticity, which judges each file's pixels separately). So all
of a case's evidence goes into a single Gemini call that returns all four
sub-verdicts as one structured JSON response — ~4x cheaper/faster than four
separate calls, and the model reasons about the whole story from one context.

Per-check applicable rules (from docs/RUBRIC.md — encoded here, not left to
the model where the rubric is absolute):
  - Relevance:    NEVER not-applicable. Forced True in code.
  - Completeness: forced False in code when the case has no video — photo-only
                  buyers must never be penalized for not filming an unboxing.
  - Tamper:       applicable only if packaging/seal is visible; the model reports
                  `packaging_visible` (a concrete perceptual fact, easier to get
                  right than an abstract "applicable" judgment).
  - Defender:     always applicable once a case reaches Rung 1b (there is always
                  evidence to argue from). It receives the Authenticity signal's
                  verdict as context so it can build on it, not re-derive it.

Direction is LOCKED: credibility 1.0 = this check finds the claim trustworthy,
0.0 = strong evidence of fraud. Failure of the call is applicable=False on all
four (NEUTRAL, never suspicion) — the caller/no-info fallback handles routing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ..contract import CaseRecord, SignalOutput
from ..vlm import VLMClient, VLMError

RUNG1B_SIGNALS = ("completeness", "tamper", "relevance", "defender")


class _SubVerdict(BaseModel):
    credibility: float = Field(description="0.0 = strong evidence of fraud on this "
                                           "check, 1.0 = this check finds the claim "
                                           "fully trustworthy")
    confidence: float = Field(description="0.0 = pure guess, 1.0 = certain")
    verdict: str = Field(description="a 1-3 word tag")
    reasoning: str = Field(description="1-2 plain sentences a non-technical "
                                       "reviewer can read")


class _TamperVerdict(_SubVerdict):
    packaging_visible: bool = Field(description="true only if parcel packaging, a "
                                                "seal, tape, or shipping wrap is "
                                                "actually visible somewhere in the "
                                                "evidence")


class _RelevanceVerdict(_SubVerdict):
    """Relevance, plus an explicit read on WHAT is actually depicted.

    `item_seen` is reported separately from `credibility` on purpose: it is the
    observation, not the judgement. Surfacing it lets the case brief say "the
    ordered item was a power bank; the video shows fabric" instead of an opaque
    0.30, and it gives a reviewer something checkable against the listing.
    """
    item_seen: str = Field(description="what the main object in the evidence "
                                       "appears to be, in a few plain words "
                                       "(e.g. 'grey fabric / clothing', 'a "
                                       "cardboard box on a floor'). Describe "
                                       "only what is visible; if nothing "
                                       "identifiable is shown, say so.")
    item_match: float = Field(description="0.0-1.0: does the object visible in "
                                          "the evidence match the ORDERED "
                                          "product named above? 1.0 = clearly "
                                          "the same product, 0.0 = clearly a "
                                          "different/unrelated object. Use 0.5 "
                                          "if the media is too unclear to tell "
                                          "— never guess.")


class _Rung1bResponse(BaseModel):
    completeness: _SubVerdict
    tamper: _TamperVerdict
    relevance: _RelevanceVerdict
    defender: _SubVerdict


_PROMPT = """You are four independent fraud-review checks for an e-commerce \
return/refund claim, executed in one pass. A buyer filed a return with the stated \
reason: "{reason}". The attached media ({n_files} file(s): {file_list}) is their \
submitted proof.
{items_line}{authenticity_line}
Evaluate the FOUR checks below INDEPENDENTLY. Do not let one check's finding bleed \
into another's score — each answers only its own question. For each check give: \
`credibility` 0.0-1.0 (1.0 = this check finds the claim trustworthy, 0.0 = strong \
evidence of fraud), `confidence` 0.0-1.0 in your own score, a 1-3 word `verdict` \
tag, and 1-2 sentences of plain `reasoning`.

CHECK 1 — COMPLETENESS: "Is the critical moment actually shown?"
  - Continuous unboxing with the box opened on camera -> credibility UP
  - No unboxing shown, or a cut/edit right at the critical moment (especially for \
"item missing" / "did not arrive" claims) -> credibility DOWN
  - If the evidence is photos only, score what the photos do show, but note that \
unboxing is not expected from photos (this check may be discarded in that case).

CHECK 2 — TAMPER: "Was the parcel already open before filming?"
  - Seal intact at the start of the video and opened on camera -> credibility UP
  - Seal already broken / box pre-opened / contents accessible before the on-camera \
opening (anything "found" could have been staged) -> credibility DOWN
  - A video that is actually a still image passed off as video (frozen or looped \
frames, no natural motion) is deception -> credibility DOWN hard
  - Also report `packaging_visible`: true only if parcel packaging, a seal, tape, \
or shipping wrap is actually visible in the evidence. If false, explain briefly \
what the evidence shows instead.

CHECK 3 — RELEVANCE: "Does the evidence depict the ORDERED ITEM, and the problem \
claimed about it?" This check has TWO parts; answer both.
  (a) `item_seen` + `item_match`: FIRST say what object the evidence actually \
shows, then compare it to the ordered product named above. Judge the OBJECT, not \
the photography — a blurry or partial shot of the right product is still a match. \
If you genuinely cannot tell what the object is, give item_match 0.5 rather than \
guessing in either direction.
  (b) `credibility`: reason says "{reason}" and the evidence shows exactly that \
problem ON THE ORDERED ITEM -> credibility UP. Evidence showing a floor, an empty \
box, an unrelated scene, or a DIFFERENT product than the one ordered -> \
credibility DOWN (this is a hard fraud signal).
  - A confident item mismatch is among the strongest signals available to you: \
damage to something the buyer did not order says nothing about this order.
  - But do NOT punish a mismatch you are unsure of. Listings are often generic, \
bundled, or multi-pack, and an accessory or a single part of the product may be \
all that is in frame. Score item_match low only when you can say what you see AND \
that it is plainly not the ordered thing.

CHECK 4 — DEFENDER: you are the not-guilty advocate. Actively hunt for SPECIFIC, \
NAMEABLE evidence that this claim is LEGITIMATE:
  - Damage patterns consistent with normal shipping/transit
  - Lighting, shadows, timestamps, and scene internally consistent
  - An intact unboxing chain with nothing staged
  - At most soft, isolated oddities rather than multiple hard red flags
  Cite the specific things you found in your reasoning. If you cannot name any \
concrete legitimacy evidence, say so and give a LOW defender credibility with low \
confidence — do NOT vaguely claim "nothing looks wrong" as if it were evidence. \
Vague reassurance is a failure; specificity is your entire job.

Calibration rules (apply to all four checks):
  - HIGH SCORES MUST BE EARNED, NOT GRANTED. A credibility above 0.7 is a claim \
that you SAW affirmative proof — you must be able to name the specific visible \
moment or detail that demonstrates it (the seal shown intact and then cut open on \
camera; one continuous unopened-to-opened take; the claimed damage clearly on the \
actual ordered item). "I found no red flags" or "nothing looks wrong" is NOT \
affirmative proof — score it 0.4-0.6, which routes the case to human review. That \
is the correct outcome for an unproven claim: this system refunds real money, and \
the fraud submissions that reach you are precisely the ones crafted to look \
plausible.
  - Reserve credibility below 0.3 for cases where you can point at concrete \
evidence of the problem the check hunts for.
  - Poor capture quality (blur, low resolution, dim light, compression, shaky \
camera) is NOT fraud and is also not proof. Honest buyers use cheap phones — never \
convert "hard to see" into a LOW score; convert it into a mid score with low \
confidence.
  - Most real submissions are mediocre-but-genuine. Judge the story, not the \
production values."""


def _items_line(rec: CaseRecord) -> str:
    """The ordered product(s), BY NAME.

    This used to emit only opaque IDs ("item 49855346760 (shop 1639973292)"),
    which meant the relevance check was asked "does the evidence depict what is
    claimed?" without ever being told what was bought. It could only compare the
    media against the return-reason string, so a video of an unrelated object
    scored on vibes. Titles were in the order data the whole time — 73 of 74
    items in test_eval have one — they simply were never put in front of the
    model. Naming the product is what makes CHECK 3's item-match question
    answerable at all.
    """
    if not rec.items:
        return ""
    parts = []
    for it in rec.items[:4]:
        title = (it.title or "").strip()
        parts.append(f'"{title}"' if title else f"item {it.item_id} (name unavailable)")
    more = f" (+{len(rec.items) - 4} more)" if len(rec.items) > 4 else ""
    return (f"The buyer ordered {len(rec.items)} item(s): "
            f"{'; '.join(parts)}{more}.\n")


def _authenticity_line(rec: CaseRecord) -> str:
    sig = rec.signals.get("authenticity")
    if sig is None or not sig.applicable:
        return ""
    return (f"Context from an earlier authenticity check (do not re-derive it): "
            f"verdict '{sig.verdict}', credibility {sig.score:.2f} — "
            f"{sig.reason_string}\n")


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _neutral(name: str, note: str) -> SignalOutput:
    return SignalOutput(signal_name=name, applicable=False, verdict="not_assessed",
                        reason_string=note, confidence=0.0)


def run(rec: CaseRecord, client: Optional[VLMClient] = None) -> dict[str, SignalOutput]:
    """One combined call -> four SignalOutputs keyed by signal name."""
    client = client or VLMClient()

    media_paths = [Path(ev.path) for ev in rec.evidence if ev.path]
    if not media_paths:
        return {n: _neutral(n, "no analyzable media") for n in RUNG1B_SIGNALS}

    file_list = ", ".join(ev.filename for ev in rec.evidence if ev.path)
    prompt = _PROMPT.format(
        reason=rec.return_reason or "unspecified",
        n_files=len(media_paths),
        file_list=file_list,
        items_line=_items_line(rec),
        authenticity_line=_authenticity_line(rec),
    )

    try:
        v: _Rung1bResponse = client.analyze(
            prompt=prompt, media=media_paths, response_schema=_Rung1bResponse,
        )
    except VLMError as e:
        note = f"combined check failed: {e}"
        return {n: _neutral(n, note) for n in RUNG1B_SIGNALS}

    out: dict[str, SignalOutput] = {}

    # Completeness — forced N/A in code for photo-only cases (rubric absolute).
    if not rec.has_video:
        out["completeness"] = SignalOutput(
            signal_name="completeness", applicable=False, verdict="photo_only",
            reason_string="Photo-only evidence; an unboxing video is not expected, "
                          "so this check is neutral by design.", confidence=0.0)
    else:
        c = v.completeness
        out["completeness"] = SignalOutput(
            signal_name="completeness", score=_clamp(c.credibility),
            verdict=c.verdict.strip(), reason_string=c.reasoning.strip(),
            confidence=_clamp(c.confidence), applicable=True)

    # Tamper — applicable only when packaging is actually visible.
    t = v.tamper
    if t.packaging_visible:
        out["tamper"] = SignalOutput(
            signal_name="tamper", score=_clamp(t.credibility),
            verdict=t.verdict.strip(), reason_string=t.reasoning.strip(),
            confidence=_clamp(t.confidence), applicable=True)
    else:
        out["tamper"] = SignalOutput(
            signal_name="tamper", applicable=False, verdict="no_packaging",
            reason_string=f"No packaging/seal visible to assess: {t.reasoning.strip()}",
            confidence=0.0)

    # Relevance — never not-applicable (rubric absolute).
    #
    # The item read is surfaced in the reason string rather than silently folded
    # into the number: a reviewer (and the case brief) can then see "ordered a
    # power bank, evidence shows grey fabric" and check it against the listing,
    # instead of being handed an unexplained 0.30. The mismatch also travels to
    # the scoring layer as its own field, so it can gate a decision on its own.
    r = v.relevance
    item_seen = (r.item_seen or "").strip()
    item_match = _clamp(r.item_match)
    reason = r.reasoning.strip()
    if item_seen:
        reason = (f"{reason} Evidence appears to show: {item_seen} "
                  f"(match to ordered item: {item_match:.2f}).")
    out["relevance"] = SignalOutput(
        signal_name="relevance", score=_clamp(r.credibility),
        verdict=r.verdict.strip(), reason_string=reason,
        confidence=_clamp(r.confidence), applicable=True,
        item_seen=item_seen or None, item_match=item_match)

    # Defender — always applicable once the case is here.
    d = v.defender
    out["defender"] = SignalOutput(
        signal_name="defender", score=_clamp(d.credibility),
        verdict=d.verdict.strip(), reason_string=d.reasoning.strip(),
        confidence=_clamp(d.confidence), applicable=True)

    return out
