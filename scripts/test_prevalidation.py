"""
Synthetic exercise of Rung 0 (sentinel.prevalidation).

No real Shopee media required — we generate images/videos on the fly (clean,
blurry, dark, tiny, corrupted, degenerate-video, and a reused-proof pair) and
assert each routes the way the design says it should. Also confirms the
placeholder thresholds at least separate obvious good from obvious bad.

Run:  python scripts/test_prevalidation.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel import db
from sentinel.contract import (
    CaseRecord, Evidence,
    DECISION_REJECT, DECISION_ESCALATE,
    REASON_DUPLICATE_PROOF, REASON_CORRUPTED_FILE,
    REASON_INSUFFICIENT_EVIDENCE, REASON_PASSED_PREVALIDATION,
)
from sentinel import prevalidation as pv

TMP = Path(tempfile.mkdtemp(prefix="rung0_"))
_rng = np.random.default_rng(42)


def _clean_image(w=640, h=480) -> np.ndarray:
    """A textured, well-exposed color image (high Laplacian variance)."""
    base = _rng.integers(40, 215, size=(h, w, 3), dtype=np.uint8)
    # add sharp edges so it's genuinely high-frequency
    cv2.rectangle(base, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (255, 255, 255), 3)
    cv2.line(base, (0, 0), (w, h), (0, 0, 0), 2)
    return base


def _write_img(name: str, arr: np.ndarray) -> str:
    p = TMP / name
    cv2.imwrite(str(p), arr)
    return str(p)


def _write_video(name: str, frames: list[np.ndarray], fps: float = 24.0) -> str:
    p = TMP / name
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return str(p)


def _case(cid: str, ev: list[Evidence]) -> CaseRecord:
    return CaseRecord(case_id=cid, session_id="S", evidence=ev)


def _img_ev(name: str, path: str) -> Evidence:
    return Evidence(filename=name, kind="image", path=path)


def _run(conn, rec) -> pv.PreValResult:
    res = pv.prevalidate(rec, conn=conn)
    pv.apply(rec, res)
    return res


PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
_failures = 0


def check(label: str, got, want) -> None:
    global _failures
    ok = got == want
    if not ok:
        _failures += 1
    print(f"  [{PASS if ok else FAIL}] {label}: got={got!r} want={want!r}")


def main() -> int:
    conn = db.connect(TMP / "test.db")
    db.init_db(conn)

    # --- clean image -> passes -------------------------------------------------
    clean = _write_img("clean.jpg", _clean_image())
    r = _run(conn, _case("C_clean", [_img_ev("clean.jpg", clean)]))
    print("Clean image:")
    check("route", r.route, None)
    check("reason", r.reason_code, REASON_PASSED_PREVALIDATION)

    # --- blurry image -> escalate / insufficient -------------------------------
    blur = _write_img("blur.jpg", cv2.GaussianBlur(_clean_image(), (31, 31), 0))
    r = _run(conn, _case("C_blur", [_img_ev("blur.jpg", blur)]))
    print("Blurry image:")
    check("route", r.route, DECISION_ESCALATE)
    check("reason", r.reason_code, REASON_INSUFFICIENT_EVIDENCE)

    # --- black image -> escalate (exposure) ------------------------------------
    dark = _write_img("dark.jpg", np.zeros((480, 640, 3), dtype=np.uint8))
    r = _run(conn, _case("C_dark", [_img_ev("dark.jpg", dark)]))
    print("Black image:")
    check("route", r.route, DECISION_ESCALATE)
    check("reason", r.reason_code, REASON_INSUFFICIENT_EVIDENCE)

    # --- tiny image -> escalate (resolution) -----------------------------------
    tiny = _write_img("tiny.jpg", _clean_image(w=120, h=90))
    r = _run(conn, _case("C_tiny", [_img_ev("tiny.jpg", tiny)]))
    print("Tiny image:")
    check("route", r.route, DECISION_ESCALATE)
    check("reason", r.reason_code, REASON_INSUFFICIENT_EVIDENCE)

    # --- corrupted file -> escalate (corrupted_file) ---------------------------
    bad = TMP / "bad.jpg"
    bad.write_bytes(b"this is not a jpeg" * 100)
    r = _run(conn, _case("C_bad", [_img_ev("bad.jpg", str(bad))]))
    print("Corrupted image:")
    check("route", r.route, DECISION_ESCALATE)
    check("reason", r.reason_code, REASON_CORRUPTED_FILE)

    # --- duplicate / reused proof -> second case REJECTS -----------------------
    dup_arr = _clean_image()
    d1 = _write_img("proofA.jpg", dup_arr)
    d2 = _write_img("proofB.jpg", dup_arr.copy())  # pixel-identical, different name/case
    r1 = _run(conn, _case("C_dupfirst", [_img_ev("proofA.jpg", d1)]))
    r2 = _run(conn, _case("C_dupsecond", [_img_ev("proofB.jpg", d2)]))
    print("Duplicate proof (first submission):")
    check("first route", r1.route, None)
    print("Duplicate proof (reused in a later case):")
    check("route", r2.route, DECISION_REJECT)
    check("reason", r2.reason_code, REASON_DUPLICATE_PROOF)
    check("names matched case", (r2.duplicate or {}).get("case_id"), "C_dupfirst")

    # --- clean short video -> passes ------------------------------------------
    good_frames = [_clean_image(w=320, h=240) for _ in range(72)]  # 3s @ 24fps
    gv = _write_video("good.mp4", good_frames)
    r = _run(conn, _case("C_goodvid", [Evidence("good.mp4", "video", gv)]))
    print("Clean video:")
    check("route", r.route, None)

    # --- degenerate (all-black) video -> escalate ------------------------------
    black_frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(48)]
    bv = _write_video("black.mp4", black_frames)
    r = _run(conn, _case("C_blackvid", [Evidence("black.mp4", "video", bv)]))
    print("All-black video:")
    check("route", r.route, DECISION_ESCALATE)
    check("reason", r.reason_code, REASON_INSUFFICIENT_EVIDENCE)

    # --- mixed evidence (one crisp + one blurry) -> PROCEEDS, not escalate -----
    good = _write_img("mix_good.jpg", _clean_image())
    soft = _write_img("mix_blur.jpg", cv2.GaussianBlur(_clean_image(), (31, 31), 0))
    r = _run(conn, _case("C_mixed", [_img_ev("mix_good.jpg", good),
                                     _img_ev("mix_blur.jpg", soft)]))
    print("Mixed evidence (1 usable + 1 blurry):")
    check("route (proceeds on the usable file)", r.route, None)

    # --- all evidence weak (both blurry, fresh content) -> escalate ------------
    w1 = _write_img("allweak1.jpg", cv2.GaussianBlur(_clean_image(), (31, 31), 0))
    w2 = _write_img("allweak2.jpg", cv2.GaussianBlur(_clean_image(), (31, 31), 0))
    r = _run(conn, _case("C_allweak", [_img_ev("allweak1.jpg", w1),
                                       _img_ev("allweak2.jpg", w2)]))
    print("All evidence weak (both blurry):")
    check("route", r.route, DECISION_ESCALATE)
    check("reason", r.reason_code, REASON_INSUFFICIENT_EVIDENCE)

    # --- absent media -> Rung 0 no-op (passes; downstream escalates) -----------
    r = _run(conn, _case("C_absent", [Evidence("missing.jpg", "image", None)]))
    print("Absent media (no pixels):")
    check("route (no-op pass)", r.route, None)

    conn.close()
    print()
    if _failures:
        print(f"\033[91m{_failures} check(s) FAILED\033[0m")
        return 1
    print("\033[92mAll Rung 0 checks passed.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
