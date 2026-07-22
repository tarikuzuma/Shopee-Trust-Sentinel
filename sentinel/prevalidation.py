"""
Rung 0 — deterministic pre-validation gate (zero API cost).

Runs BEFORE the five VLM checks. Uses only cv2 / Pillow / numpy / scipy, so it
answers the judge's stated bottleneck (GPU/API cost at scale) before a single
model call happens. Every stage that hard-stops means the expensive agents never
run for that case.

ROUTING PHILOSOPHY (locked with the user 2026-07-22):
  Quality failure is NOT fraud. A blurry / dark / low-res / too-short clip from a
  cheap phone is an honest buyer we must protect, not auto-reject. So the failure
  classes route differently:

    - Corrupted / won't decode        -> ESCALATE  (corrupted_file)      ops failure
    - Too low-res / blurry / dark /
      too short / degenerate          -> ESCALATE  (insufficient_evidence)
    - Duplicate / reused proof        -> REJECT    (duplicate_proof)      FRAUD
    - EXIF editing fingerprint        -> pass through as a PRIOR into Authenticity

  Only the duplicate-hash match is dispositive fraud (like the scoring layer's
  strong-fraud veto): it short-circuits the whole pipeline straight to reject.
  Everything else that "fails" escalates with a reason code so the swipe app can
  bounce it for resubmission without doing case analysis. This honors the core
  invariant: an unjudgeable signal is NEUTRAL / escalate, never suspicion.

Frozen/looped-frame ("a still image passed off as video") is deliberately NOT
owned here — it belongs to the Tamper agent, because a loop is evidence of
deception ("was faked"), a stronger claim than Rung 0's "can't be judged."

THRESHOLD DISCIPLINE: every numeric floor below is a PLACEHOLDER to be fit against
the labeled dataset (same discipline as the 35/75 credibility thresholds), NOT
asserted from intuition. They are marked `# TODO(tune)`. Do not trust the demo
numbers until they've been swept over real Shopee media — low-end-device photos
are a large fraction of real traffic, and a mis-set floor rejects valid ones.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .contract import (
    CaseRecord,
    Evidence,
    DECISION_REJECT,
    DECISION_ESCALATE,
    REASON_DUPLICATE_PROOF,
    REASON_CORRUPTED_FILE,
    REASON_INSUFFICIENT_EVIDENCE,
    REASON_PASSED_PREVALIDATION,
)

# ---------------------------------------------------------------------------
# Thresholds — ALL placeholders, fit against the labeled dataset before trusting.
# ---------------------------------------------------------------------------
MIN_FILE_BYTES = 1024                 # TODO(tune): reject 0-byte / absurdly tiny files pre-decode
MIN_RESOLUTION_SHORT_EDGE = 240       # TODO(tune): px on the SHORT edge below which a
                                      #   "smashed vs intact item" distinction is impossible
BLUR_REF_LONG_EDGE = 512              # normalize sharpness across cameras: resize so the
                                      #   long edge = this BEFORE computing Laplacian variance
BLUR_LAPLACIAN_VAR_MIN = 40.0         # TODO(tune): variance-of-Laplacian floor (post-resize)
BRIGHTNESS_MIN = 25.0                 # TODO(tune): mean grayscale brightness floor (0-255)
BRIGHTNESS_MAX = 235.0                # TODO(tune): mean grayscale brightness ceiling
CLIP_FRACTION_MAX = 0.60              # TODO(tune): max fraction of pixels crushed to 0 or 255
VIDEO_MIN_DURATION_S = 1.5            # TODO(tune): physical floor for an unboxing clip
VIDEO_FRAME_SAMPLES = 10              # evenly spaced frames sampled for video metrics
DEGENERATE_FRAME_STD_MAX = 6.0        # TODO(tune): per-frame stddev below which a frame is
                                      #   effectively blank (lens cap / ceiling / black)

# Perceptual-hash duplicate gate. Distance is Hamming over the 64-bit pHash.
HASH_MAX_DISTANCE = 6                 # TODO(tune): <= this == near-identical == reused proof

# EXIF fields whose presence indicates an image passed through editing/AI tooling.
# Presence is a real signal (a prior for Authenticity); ABSENCE is neutral — many
# apps and all screenshots strip EXIF, so absence must never be penalized.
_EDIT_SOFTWARE_MARKERS = (
    "photoshop", "gimp", "lightroom", "affinity", "snapseed", "picsart",
    "midjourney", "stable diffusion", "dall-e", "dall e", "firefly", "canva",
)


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------
@dataclass
class PreValResult:
    """Outcome of Rung 0 for one case.

    `route` is None when the case CLEARS the gate and should proceed to the five
    VLM agents. Otherwise it is a terminal decision (reject/escalate) and the
    agents are skipped.
    """
    route: Optional[str] = None                 # None => proceed; else reject/escalate
    reason_code: str = REASON_PASSED_PREVALIDATION
    evidence_qc: list[dict] = field(default_factory=list)   # per-file metrics + verdict
    exif_priors: dict = field(default_factory=dict)         # filename -> editing-software marker
    duplicate: Optional[dict] = None            # {case_id, filename, distance} if reused
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.route is None

    def to_dict(self) -> dict:
        return {
            "route": self.route,
            "reason_code": self.reason_code,
            "evidence_qc": self.evidence_qc,
            "exif_priors": self.exif_priors,
            "duplicate": self.duplicate,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Perceptual hash (DCT-based, imagehash-compatible) — no imagehash dependency.
# ---------------------------------------------------------------------------
def phash_from_gray(gray: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """8x8 DCT perceptual hash of a grayscale image, returned as 16-char hex."""
    from scipy.fftpack import dct
    import cv2

    img_size = hash_size * highfreq_factor
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    pixels = resized.astype(np.float64)
    coeffs = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coeffs[:hash_size, :hash_size]
    med = np.median(low)
    bits = (low > med).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:016x}"


# ---------------------------------------------------------------------------
# Stage helpers — operate on a grayscale ndarray so image + video share code.
# ---------------------------------------------------------------------------
def _normalized_laplacian_var(gray: np.ndarray) -> float:
    """Variance of Laplacian, normalized by resizing the long edge to a fixed
    reference first. Without this, the metric scales with resolution and any
    single threshold is meaningless across different cameras."""
    import cv2

    h, w = gray.shape[:2]
    long_edge = max(h, w)
    if long_edge > BLUR_REF_LONG_EDGE:
        scale = BLUR_REF_LONG_EDGE / float(long_edge)
        gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _exposure_metrics(gray: np.ndarray) -> tuple[float, float]:
    """(mean brightness 0-255, fraction of pixels clipped at 0 or 255)."""
    mean = float(gray.mean())
    total = gray.size
    clipped = int(np.count_nonzero(gray <= 1) + np.count_nonzero(gray >= 254))
    return mean, clipped / float(total) if total else 0.0


def _quality_verdict(width: int, height: int, sharp: float,
                     brightness: float, clip_frac: float) -> Optional[str]:
    """Return an insufficient-* note if a quality gate fails, else None."""
    if min(width, height) < MIN_RESOLUTION_SHORT_EDGE:
        return f"insufficient_resolution ({width}x{height})"
    if sharp < BLUR_LAPLACIAN_VAR_MIN:
        return f"insufficient_sharpness (lapvar={sharp:.1f})"
    if not (BRIGHTNESS_MIN <= brightness <= BRIGHTNESS_MAX):
        return f"insufficient_exposure (mean={brightness:.1f})"
    if clip_frac > CLIP_FRACTION_MAX:
        return f"insufficient_exposure (clipped={clip_frac:.0%})"
    return None


# ---------------------------------------------------------------------------
# Per-file evaluation
# ---------------------------------------------------------------------------
def _read_exif_editor(path: Path) -> Optional[str]:
    """Return an editing-software marker string if EXIF exposes one, else None.
    Absence is neutral and must never be penalized."""
    try:
        from PIL import Image
        img = Image.open(str(path))
        exif = getattr(img, "getexif", lambda: None)()
        if not exif:
            return None
        # 0x0131 = Software tag; also scan any string value for known markers.
        haystacks = []
        soft = exif.get(0x0131)
        if soft:
            haystacks.append(str(soft))
        for v in exif.values():
            if isinstance(v, str):
                haystacks.append(v)
        blob = " ".join(haystacks).lower()
        for marker in _EDIT_SOFTWARE_MARKERS:
            if marker in blob:
                return marker
    except Exception:
        return None
    return None


def _eval_image(path: Path) -> dict:
    """Timed wrapper around image evaluation — stamps eval_ms onto the metrics."""
    t0 = time.perf_counter()
    m = _eval_image_impl(path)
    m["eval_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    return m


def _eval_image_impl(path: Path) -> dict:
    """Decode + QC + pHash one image. Returns a metrics dict with 'status'."""
    import cv2

    if path.stat().st_size < MIN_FILE_BYTES:
        return {"kind": "image", "status": "corrupted", "note": "file too small"}
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {"kind": "image", "status": "corrupted", "note": "cv2 could not decode"}
    h, w = gray.shape[:2]
    sharp = _normalized_laplacian_var(gray)
    brightness, clip_frac = _exposure_metrics(gray)
    std = float(gray.std())
    # A perceptual hash of near-blank content is NOT a real fingerprint: every
    # blank/degenerate frame hashes to ~the same value, so hashing them would
    # manufacture false "reused proof" rejects between unrelated honest buyers.
    # Only hash media that has actual visual content.
    phash = phash_from_gray(gray) if std >= DEGENERATE_FRAME_STD_MAX else None
    quality_note = _quality_verdict(w, h, sharp, brightness, clip_frac)
    return {
        "kind": "image", "width": w, "height": h,
        "sharpness": round(sharp, 1), "brightness": round(brightness, 1),
        "clip_fraction": round(clip_frac, 3), "frame_std": round(std, 1),
        "phash": phash,
        "status": "insufficient" if quality_note else "ok",
        "note": quality_note or "",
    }


def _eval_video(path: Path) -> dict:
    """Timed wrapper around video evaluation — stamps eval_ms onto the metrics."""
    t0 = time.perf_counter()
    m = _eval_video_impl(path)
    m["eval_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    return m


def _eval_video_impl(path: Path) -> dict:
    """Decode + sample frames + QC + pHash one video (median over sampled frames)."""
    import cv2

    if path.stat().st_size < MIN_FILE_BYTES:
        return {"kind": "video", "status": "corrupted", "note": "file too small"}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"kind": "video", "status": "corrupted", "note": "cv2 could not open"}
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frame_count / fps) if fps > 0 else 0.0

        if frame_count <= 0 or w == 0 or h == 0:
            return {"kind": "video", "status": "corrupted", "note": "no decodable frames"}

        idxs = np.linspace(0, frame_count - 1, num=min(VIDEO_FRAME_SAMPLES, frame_count),
                           dtype=int)
        sharps, brights, clips, stds, mid_phash = [], [], [], [], None
        for i, fi in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharps.append(_normalized_laplacian_var(gray))
            b, c = _exposure_metrics(gray)
            brights.append(b)
            clips.append(c)
            stds.append(float(gray.std()))
            if i == len(idxs) // 2:          # representative frame for duplicate hashing
                mid_phash = phash_from_gray(gray)
    finally:
        cap.release()

    if not sharps:
        return {"kind": "video", "status": "corrupted", "note": "frames unreadable"}

    # Median, not mean/first: one motion-blurred frame shouldn't kill a clear video.
    sharp = float(np.median(sharps))
    brightness = float(np.median(brights))
    clip_frac = float(np.median(clips))
    frame_std = float(np.median(stds))

    metrics = {
        "kind": "video", "width": w, "height": h, "duration_s": round(duration, 2),
        "sharpness": round(sharp, 1), "brightness": round(brightness, 1),
        "clip_fraction": round(clip_frac, 3), "frame_std": round(frame_std, 1),
        # Suppress the hash for degenerate/blank video — see _eval_image note.
        "phash": mid_phash if frame_std >= DEGENERATE_FRAME_STD_MAX else None,
    }

    # Degeneracy: essentially blank frames (lens cap, ceiling, all-black).
    if frame_std < DEGENERATE_FRAME_STD_MAX:
        metrics.update(status="insufficient", note=f"degenerate/blank frames (std={frame_std:.1f})")
        return metrics
    if duration < VIDEO_MIN_DURATION_S:
        metrics.update(status="insufficient", note=f"too short ({duration:.2f}s)")
        return metrics
    quality_note = _quality_verdict(w, h, sharp, brightness, clip_frac)
    metrics.update(status="insufficient" if quality_note else "ok", note=quality_note or "")
    return metrics


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def prevalidate(rec: CaseRecord, conn=None, session_id: Optional[str] = None,
                record: bool = True) -> PreValResult:
    """Run Rung 0 over a case's evidence and decide its route.

    conn (optional): a sqlite connection enabling cross-case duplicate detection.
      Without it, every pixel gate still runs; only the duplicate check is skipped
      (useful for unit tests). When present, each cleared file's pHash is recorded
      to the running store (unless record=False) so LATER cases can match it.

    Precedence: duplicate (reject) > corrupted/quality (escalate) > pass. A
    duplicate short-circuits the whole case immediately — a hash already proved
    reuse, so we don't waste a VLM call confirming it, nor let a "blurry" verdict
    downgrade it to a mere escalate.
    """
    from . import db

    sid = session_id or rec.session_id
    result = PreValResult()

    escalate_reason: Optional[str] = None  # a quality/corruption fail, if no dup found
    has_usable = False                     # at least one 'ok' evidence file present

    for ev in rec.evidence:
        if not ev.path:
            # No pixels to inspect (media not in hand). Rung 0 is a no-op for this
            # file — it must NOT manufacture a failure; downstream no_information
            # handles a case with nothing to judge. See media-files-blocker.
            result.evidence_qc.append({"filename": ev.filename, "status": "absent"})
            continue

        path = Path(ev.path)
        kind = ev.kind if ev.kind in ("image", "video") else Evidence.infer_kind(ev.filename)
        metrics = _eval_video(path) if kind == "video" else _eval_image(path)
        metrics["filename"] = ev.filename
        result.evidence_qc.append(metrics)

        # Stage A — file integrity (this FILE fails; case-level route decided below).
        if metrics["status"] == "corrupted":
            escalate_reason = escalate_reason or REASON_CORRUPTED_FILE
            continue  # can't hash a file we couldn't decode

        phash = metrics.get("phash")
        if phash:
            ev.phash = phash
            # Stage F — perceptual duplicate gate (fast-track REJECT, short-circuit).
            if conn is not None:
                dup = db.find_duplicate_phash(conn, phash, sid, rec.case_id, HASH_MAX_DISTANCE)
                if dup is not None:
                    result.route = DECISION_REJECT
                    result.reason_code = REASON_DUPLICATE_PROOF
                    result.duplicate = dup
                    result.notes.append(
                        f"{ev.filename} reuses proof from case {dup['case_id']} "
                        f"({dup['filename']}, hamming={dup['distance']})"
                    )
                    return result  # dispositive — skip the agents entirely

        # Stage G — EXIF editing/AI fingerprint (SOFT prior, never a gate).
        if kind == "image":
            editor = _read_exif_editor(path)
            if editor:
                ev.exif_editor = editor
                result.exif_priors[ev.filename] = editor

        # Stages B-E — quality gates. A weak FILE is noted, but does not by itself
        # sink the case (see case-level roll-up below).
        if metrics["status"] == "insufficient":
            escalate_reason = escalate_reason or REASON_INSUFFICIENT_EVIDENCE
        else:  # status == "ok"
            has_usable = True

    # No duplicate found. Record cleared hashes so later cases can match this one.
    if conn is not None and record:
        for m in result.evidence_qc:
            if m.get("status") in ("ok", "insufficient") and m.get("phash"):
                db.record_phash(conn, sid, rec.case_id, m["filename"], m["phash"])

    # Case-level roll-up: proceed if ANY evidence file is usable — the agents can
    # judge on it, and a single weak/corrupt file must not force a human review
    # (raises automation; still safe — quality != fraud, and a duplicate already
    # returned above). Escalate only when something failed AND nothing is usable.
    # All-absent falls through (route stays None) to downstream no_information.
    if not has_usable and escalate_reason is not None:
        result.route = DECISION_ESCALATE
        result.reason_code = escalate_reason

    return result


def apply(rec: CaseRecord, result: PreValResult) -> bool:
    """Write a Rung-0 result onto the case. Returns True if the gate produced a
    TERMINAL decision (agents should be SKIPPED), False if the case should proceed
    to the five VLM checks.

    On a pass, the trace (incl. EXIF priors) is still attached so the Authenticity
    agent can read exif_priors and the brief can explain that Rung 0 was clean.
    """
    rec.prevalidation = result.to_dict()
    if result.passed:
        return False
    rec.decision = result.route
    rec.reason_code = result.reason_code
    return True
