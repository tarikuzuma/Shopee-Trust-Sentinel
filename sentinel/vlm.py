"""
VLM provider abstraction.

The checks talk to this, never to a vendor SDK directly, so swapping Gemini for
Claude later is a one-class change. Primary provider is Gemini Flash: it takes
video natively (via the Files API) and its speed/cost fit the judge's stated
GPU-cost-at-scale bottleneck.

Design:
  - Images are sent INLINE (fast, no upload round-trip).
  - Videos are uploaded via the Files API (they exceed the inline request cap and
    need server-side processing), then referenced by handle.
  - Structured output: we pass a response schema and read back parsed JSON, so
    callers get a typed object, not free text to regex.
  - The client never invents a verdict on failure — it raises VLMError, and the
    caller decides (for us: a failed authenticity call is NEUTRAL, never fraud).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Sequence

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # read .env (git-ignored) into the environment

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

_MIME = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
    "webp": "image/webp", "gif": "image/gif", "bmp": "image/bmp",
    "mp4": "video/mp4", "mov": "video/quicktime", "avi": "video/x-msvideo",
    "mkv": "video/x-matroska", "webm": "video/webm",
}

# Videos this size or smaller can be sent inline; larger ones must use the Files
# API. 18 MB leaves headroom under the ~20 MB total-request cap for the prompt.
_INLINE_VIDEO_MAX_BYTES = 18 * 1024 * 1024
_UPLOAD_POLL_SECONDS = 1.0
_UPLOAD_TIMEOUT_SECONDS = 120.0


class VLMError(Exception):
    """A provider call failed. Callers must treat this as NO SIGNAL (neutral)."""


def _mime_for(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    if ext not in _MIME:
        raise VLMError(f"unsupported media type: {path.name}")
    return _MIME[ext]


class VLMClient:
    """Gemini-backed multimodal client. Construct once, reuse across cases."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise VLMError("GEMINI_API_KEY not set (put it in .env or the env).")
        self.model = model or DEFAULT_MODEL
        self._client = genai.Client(api_key=key)
        # Upload cache: the same video is used by both the Authenticity call and
        # the Rung 1b call — re-uploading it would double time and quota for no
        # benefit. Keyed by resolved path + mtime so a changed file re-uploads.
        self._upload_cache: dict[str, object] = {}
        # Usage metadata of the most recent analyze() call (for cost accounting).
        self.last_usage = None

    # --- media handling ------------------------------------------------------

    def _media_part(self, path: Path):
        """Turn one media file into a content part (inline image, or uploaded video)."""
        mime = _mime_for(path)
        size = path.stat().st_size
        if mime.startswith("video/") or size > _INLINE_VIDEO_MAX_BYTES:
            return self._upload(path)
        return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)

    def _upload(self, path: Path):
        """Upload via the Files API and block until the file is ACTIVE (cached)."""
        cache_key = f"{path.resolve()}:{path.stat().st_mtime_ns}"
        cached = self._upload_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            f = self._client.files.upload(file=str(path))
        except Exception as e:  # noqa: BLE001
            raise VLMError(f"upload failed for {path.name}: {e}") from e
        deadline = time.time() + _UPLOAD_TIMEOUT_SECONDS
        while getattr(f.state, "name", str(f.state)) == "PROCESSING":
            if time.time() > deadline:
                raise VLMError(f"upload processing timed out for {path.name}")
            time.sleep(_UPLOAD_POLL_SECONDS)
            f = self._client.files.get(name=f.name)
        if getattr(f.state, "name", str(f.state)) == "FAILED":
            raise VLMError(f"server failed to process {path.name}")
        self._upload_cache[cache_key] = f
        return f

    # --- the one call the checks use -----------------------------------------

    def analyze(self, prompt: str, media: Sequence[Path],
                response_schema, temperature: float = 0.0):
        """Send prompt + media, return the parsed structured response.

        `response_schema` is a Pydantic model class; the return is an instance of
        it. Raises VLMError on any failure (network, parse, unsupported media) —
        the caller must map that to a neutral, non-suspicious signal.
        """
        parts: list = [prompt]
        for m in media:
            parts.append(self._media_part(Path(m)))

        cfg = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
        )
        try:
            resp = self._client.models.generate_content(
                model=self.model, contents=parts, config=cfg,
            )
        except Exception as e:  # noqa: BLE001
            raise VLMError(f"generate_content failed: {e}") from e

        self.last_usage = getattr(resp, "usage_metadata", None)
        parsed = getattr(resp, "parsed", None)
        if parsed is None:
            raise VLMError("model returned no parseable structured output")
        return parsed
