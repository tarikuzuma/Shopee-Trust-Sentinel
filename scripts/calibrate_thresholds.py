"""
Rung 0 threshold calibration — replaces intuition with the dataset's own numbers.

Scans every media file under media/, runs the SAME per-file evaluation Rung 0
uses (decode + resolution + normalized sharpness + exposure + duration + frame
std), and prints the distribution of each metric plus how many files the CURRENT
placeholder thresholds would gate. Use it to set each floor from data — the
rubric's "threshold discipline" made executable.

This is a calibration tool, not a gate: it never rejects anything. Run it again
after unzipping the full test media for a bigger sample.

Run:  python scripts/calibrate_thresholds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from sentinel import prevalidation as pv
from sentinel.media import MEDIA_DIR
from sentinel.contract import Evidence

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _pctl(vals: list[float], p: float) -> float:
    return float(np.percentile(vals, p)) if vals else float("nan")


def _dist(name: str, vals: list[float], floor=None, ceil=None) -> None:
    if not vals:
        print(f"  {name:16} (no data)")
        return
    v = sorted(vals)
    line = (f"  {name:16} n={len(v):3}  min={v[0]:8.1f}  p05={_pctl(v,5):8.1f}  "
            f"med={_pctl(v,50):8.1f}  p95={_pctl(v,95):8.1f}  max={v[-1]:8.1f}")
    if floor is not None:
        n_below = sum(1 for x in v if x < floor)
        line += f"   [<{floor} would gate {n_below}/{len(v)}]"
    if ceil is not None:
        n_above = sum(1 for x in v if x > ceil)
        line += f"   [>{ceil} would gate {n_above}/{len(v)}]"
    print(line)


def main() -> int:
    files = [p for p in MEDIA_DIR.rglob("*")
             if p.is_file() and p.suffix.lower() in (IMG_EXT | VID_EXT)]
    if not files:
        print(f"No media found under {MEDIA_DIR}. Drop files there first.")
        return 1

    print(f"Scanning {len(files)} media files under {MEDIA_DIR}\n")

    short_edge, sharp, bright, clipf, dur, fstd = [], [], [], [], [], []
    n_corrupt = 0
    gated = {"resolution": [], "sharpness": [], "exposure": [], "short_video": []}

    for p in files:
        kind = "video" if p.suffix.lower() in VID_EXT else "image"
        m = pv._eval_video(p) if kind == "video" else pv._eval_image(p)
        if m["status"] == "corrupted":
            n_corrupt += 1
            continue
        w, h = m.get("width", 0), m.get("height", 0)
        short_edge.append(min(w, h))
        sharp.append(m["sharpness"])
        bright.append(m["brightness"])
        clipf.append(m["clip_fraction"] * 100.0)
        if kind == "video":
            dur.append(m.get("duration_s", 0.0))
            fstd.append(m.get("frame_std", 0.0))
        # which current gate (if any) this file trips
        note = m.get("note", "")
        if "resolution" in note:
            gated["resolution"].append(p.name)
        elif "sharpness" in note:
            gated["sharpness"].append(p.name)
        elif "exposure" in note:
            gated["exposure"].append(p.name)
        elif "short" in note:
            gated["short_video"].append(p.name)

    print("METRIC DISTRIBUTIONS  (current placeholder floor/ceiling shown in [])")
    _dist("short_edge_px", short_edge, floor=pv.MIN_RESOLUTION_SHORT_EDGE)
    _dist("sharpness", sharp, floor=pv.BLUR_LAPLACIAN_VAR_MIN)
    _dist("brightness", bright, floor=pv.BRIGHTNESS_MIN, ceil=pv.BRIGHTNESS_MAX)
    _dist("clip_pct", clipf, ceil=pv.CLIP_FRACTION_MAX * 100.0)
    _dist("duration_s", dur, floor=pv.VIDEO_MIN_DURATION_S)
    _dist("frame_std", fstd, floor=pv.DEGENERATE_FRAME_STD_MAX)

    print(f"\ncorrupted/undecodable: {n_corrupt}/{len(files)}")
    print("\nFILES THE CURRENT THRESHOLDS WOULD GATE (sanity-check these are truly bad):")
    any_gated = False
    for gate, names in gated.items():
        if names:
            any_gated = True
            print(f"  {gate:12} -> {names}")
    if not any_gated:
        print("  (none)")

    print("\nHOW TO USE: a metric whose p05 sits just ABOVE its floor is at risk of "
          "\n  clipping valid low-end-device media - lower the floor toward the p05 of "
          "\n  media you believe is genuine. Re-run after unzipping the full test set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
