"""
Authenticity check (Rung 1a) — "is the media manipulated?"

The hardest, highest-value check (the one the judge's own team couldn't crack),
so it runs SOLO and FIRST. If it alone meets the strong-fraud veto bar, the case
rejects and the other checks never run (that short-circuit lives in the
orchestrator, not here — this module only produces the SignalOutput).

Scope (owns): full-frame AI generation, local splice/edit, physical/compositional
implausibility, recapture/screenshot. Does NOT own: relevance to the return
reason (Relevance), whether the parcel was opened (Tamper), or frozen/looped
frames (Tamper). It judges the authenticity of the pixels, nothing else.

Direction is LOCKED: credibility 1.0 = clearly authentic, 0.0 = clearly fake.
Aggregation across multiple evidence files is WORST-CASE (min credibility) — one
faked photo among honest ones is still fraud.

Failure is neutral, never suspicion: if the VLM call errors or there is no usable
media, the signal is applicable=False (contributes NEUTRAL). We never manufacture
a fraud verdict from a missing signal.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ..contract import CaseRecord, SignalOutput
from ..vlm import VLMClient, VLMError

SIGNAL_NAME = "authenticity"


class _Verdict(BaseModel):
    """Structured response for one evidence file."""
    credibility: float = Field(description="0.0 = clearly AI-generated/edited, "
                                           "1.0 = clearly an authentic camera capture")
    confidence: float = Field(description="0.0 = pure guess, 1.0 = certain of the "
                                          "credibility score above")
    verdict: str = Field(description="a 1-3 word tag, e.g. 'authentic', "
                                     "'ai-generated', 'spliced', 'recaptured screen'")
    reasoning: str = Field(description="one or two plain sentences a non-technical "
                                       "reviewer can read")


_PROMPT = """You are a forensic media-authenticity check for an e-commerce return-\
fraud system. A buyer submitted this {kind} as proof for a return claim.

Judge ONLY whether the media is an authentic, unmanipulated capture. Do NOT judge \
whether it matches the return reason, whether a parcel was opened, or whether the \
damage is the seller's fault — other checks handle those.

Return-reason context (for understanding the scene only, NOT to be graded): {reason}
{exif_line}
Score `credibility` from 0.0 to 1.0:
  1.0 = clearly a real photo/video straight from a camera
  0.0 = clearly AI-generated or digitally manipulated

LOWER credibility for genuine manipulation signals:
  - AI-generation artifacts: warped/garbled text or logos, impossible geometry, \
melted or repeated textures, too-smooth surfaces, nonsensical fine detail
  - Splicing/editing: mismatched lighting or shadows between regions, inconsistent \
noise/compression across the frame, cloned patches, unnatural edges
  - Physically implausible damage (cracks, spills, breakage that could not occur)
  - Recapture: a photo of a screen showing another image (moire, screen glare, \
bezel, pixel grid)

Do NOT lower credibility for ordinary poor capture quality: low resolution, JPEG \
blockiness, motion blur, dim lighting, sensor noise. Honest buyers use cheap phone \
cameras — poor quality is NOT manipulation.

Calibration: most real buyers submit mediocre-but-genuine media. Reserve \
credibility below 0.3 for real evidence of generation or editing. If you are only \
uncertain, stay near 0.5 and report LOW confidence. Never treat "looks low quality" \
as "looks fake"."""


def _prompt_for(rec: CaseRecord, kind: str, exif_editor: Optional[str]) -> str:
    exif_line = ""
    if exif_editor:
        exif_line = (f"\nMetadata note (a prior, not proof): this image's EXIF names "
                     f"editing software '{exif_editor}'. Weigh it, but many benign "
                     f"apps also write this tag — do not treat it as decisive.\n")
    return _PROMPT.format(kind=kind, reason=rec.return_reason or "unspecified",
                          exif_line=exif_line)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def run(rec: CaseRecord, client: Optional[VLMClient] = None) -> SignalOutput:
    """Assess authenticity of a case's media and return the SignalOutput.

    One VLM call per evidence file; worst-case (min credibility) wins. Files with
    no local path are skipped. If nothing could be assessed, the signal is
    applicable=False (neutral).
    """
    client = client or VLMClient()

    per_file: list[tuple[float, float, str, str, str]] = []  # cred, conf, verdict, reason, name
    errors: list[str] = []

    for ev in rec.evidence:
        if not ev.path:
            continue
        kind = ev.kind if ev.kind in ("image", "video") else "image"
        try:
            v: _Verdict = client.analyze(
                prompt=_prompt_for(rec, kind, ev.exif_editor),
                media=[Path(ev.path)],
                response_schema=_Verdict,
            )
        except VLMError as e:
            errors.append(f"{ev.filename}: {e}")
            continue
        per_file.append((_clamp(v.credibility), _clamp(v.confidence),
                         v.verdict.strip(), v.reasoning.strip(), ev.filename))

    if not per_file:
        # Nothing assessable — NEUTRAL, never suspicion. Note the error for the trace.
        note = "no analyzable media" if not errors else f"all checks failed: {'; '.join(errors)}"
        return SignalOutput(signal_name=SIGNAL_NAME, applicable=False,
                            verdict="not_assessed", reason_string=note, confidence=0.0)

    # Worst-case wins: the least-credible file drives the verdict.
    per_file.sort(key=lambda t: t[0])
    cred, conf, verdict, reason, name = per_file[0]

    if len(per_file) > 1:
        reason = (f"Worst of {len(per_file)} files ({name}): {reason}")
    if errors:
        reason += f"  [note: {len(errors)} file(s) could not be analyzed]"

    return SignalOutput(
        signal_name=SIGNAL_NAME,
        score=cred,
        verdict=verdict,
        reason_string=reason,
        confidence=conf,
        applicable=True,
    )
