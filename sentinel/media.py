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

# filename -> path index, built once by scanning media/ recursively. This lets
# the user drop files flat OR in train/test subfolders — either layout works.
_INDEX: Optional[dict[str, Path]] = None


def _build_index() -> dict[str, Path]:
    idx: dict[str, Path] = {}
    if MEDIA_DIR.exists():
        for p in MEDIA_DIR.rglob("*"):
            if p.is_file() and p.name.lower() != "readme.md":
                # First match wins; Order-ID filenames are globally unique so
                # collisions across subfolders are not expected.
                idx.setdefault(p.name, p)
                idx.setdefault(p.name.strip(), p)
    return idx


def refresh_index() -> None:
    """Rebuild the index (call after adding files mid-session)."""
    global _INDEX
    _INDEX = _build_index()


def resolve(filename: str) -> Optional[Path]:
    """Return the on-disk path for an evidence filename, or None if absent."""
    global _INDEX
    if _INDEX is None:
        _INDEX = _build_index()
    return _INDEX.get(filename) or _INDEX.get(filename.strip())


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
