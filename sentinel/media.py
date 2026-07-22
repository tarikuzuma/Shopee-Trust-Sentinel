"""
Resolves evidence filenames to on-disk paths under media/.

A missing file is not an error — the corresponding VLM checks simply return
applicable=False (NEUTRAL), honoring the rule that a missing signal never lowers
credibility.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .contract import CaseRecord

MEDIA_DIR = Path(__file__).resolve().parent.parent / "media"


def resolve(filename: str) -> Optional[Path]:
    """Return the on-disk path for an evidence filename, or None if absent."""
    p = MEDIA_DIR / filename
    if p.exists():
        return p
    # Tolerate stray leading/trailing spaces in the sheet's filenames.
    stripped = MEDIA_DIR / filename.strip()
    if stripped.exists():
        return stripped
    return None


def attach_paths(rec: CaseRecord) -> CaseRecord:
    """Populate Evidence.path for every present media file on the case."""
    for ev in rec.evidence:
        hit = resolve(ev.filename)
        ev.path = str(hit) if hit else None
    return rec


def present_count(rec: CaseRecord) -> tuple[int, int]:
    """(files present, total referenced) — handy for a pre-run readiness check."""
    total = len(rec.evidence)
    present = sum(1 for ev in rec.evidence if resolve(ev.filename) is not None)
    return present, total
