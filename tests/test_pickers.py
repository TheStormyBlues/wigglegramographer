#!/usr/bin/env python3
"""Tests for the interactive pickers and their coordinate maths.

The picker windows cannot be driven by hand in a test, so the highgui calls are
stubbed and the state machine is fed synthetic mouse events. This covers the
parts that break silently: the pixel <-> fraction mapping, drag normalisation,
clamping, the reject-a-stray-click gate, and the no-GUI fallback.

    python tests/test_pickers.py        # exits non-zero if anything fails
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import wigglegram as wg

FAILURES = []


def check(label, cond, detail=""):
    ok = bool(cond)
    if not ok:
        FAILURES.append(label)
    print(f"   {'ok  ' if ok else 'FAIL'}  {label}{('   ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# Stub harness
# ---------------------------------------------------------------------------
_REAL = {n: getattr(cv2, n) for n in
         ("namedWindow", "setMouseCallback", "imshow", "waitKey",
          "getWindowProperty", "destroyWindow", "destroyAllWindows")}

DOWN, MOVE, UP = cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP
ENTER, SPACE, ESC = 13, 32, 27


def drive(fn, script, fail_window=False):
    """Run `fn` with highgui stubbed, applying one (events, key) step per frame."""
    state = {"cb": None, "step": 0, "shown": 0}

    def named(*a, **k):
        if fail_window:
            raise cv2.error("no gui")

    def waitkey(_):
        i = state["step"]
        state["step"] += 1
        if i >= len(script):
            return ESC                      # backstop against a runaway loop
        events, key = script[i]
        for ev, x, y in events:
            state["cb"](ev, x, y, 0, None)
        return key

    def imshow(_w, img):
        state["shown"] += 1

    for name, f in (("namedWindow", named),
                    ("setMouseCallback", lambda w, cb, *a: state.__setitem__("cb", cb)),
                    ("imshow", imshow), ("waitKey", waitkey),
                    ("getWindowProperty", lambda *a: 1.0),
                    ("destroyWindow", lambda *a: None),
                    ("destroyAllWindows", lambda *a: None)):
        setattr(cv2, name, f)
    try:
        return fn()
    finally:
        for name, f in _REAL.items():
            setattr(cv2, name, f)


# ---------------------------------------------------------------------------
# Anchor box: pixel <-> fraction mapping
# ---------------------------------------------------------------------------
print("anchor mapping")
FRAME_H, FRAME_W = 2276, 1809
DEFAULT = (0.5, 0.6, 0.45, 0.55)

seen = []
for dw, dh in ((FRAME_W, FRAME_H), (715, 900), (318, 400)):
    box = [int(0.30 * dw), int(0.20 * dh), int(0.62 * dw), int(0.55 * dh)]
    seen.append(wg._box_to_anchor(box, dw, dh))
spread = max(max(abs(v[i] - seen[0][i]) for i in range(4)) for v in seen)
check("fractions are independent of preview scale", spread < 0.005, f"spread {spread:.5f}")

back = wg._box_to_anchor(wg._anchor_to_box(DEFAULT, 715, 900), 715, 900)
err = max(abs(a - b) for a, b in zip(DEFAULT, back))
check("anchor round-trips through pixel space", err < 0.005, f"max err {err:.5f}")

fwd = wg._box_to_anchor([100, 120, 400, 500], 715, 900)
rev = wg._box_to_anchor([400, 500, 100, 120], 715, 900)
check("inverted drag normalises", fwd == rev)

over = wg._box_to_anchor([-50, -80, 9999, 9999], 715, 900)
check("overrun drag clamps to the frame", over == (0.5, 0.5, 1.0, 1.0), str(over))

# ---------------------------------------------------------------------------
# Anchor picker state machine
# ---------------------------------------------------------------------------
print("anchor picker")
FRAME = np.random.randint(0, 255, (FRAME_H, FRAME_W, 3), np.uint8)
DW, DH = 715, 900
drag = [(DOWN, 200, 300), (MOVE, 400, 550), (UP, 500, 700)]
want = (350 / DW, 500 / DH, 300 / DW, 400 / DH)


def pick():
    return wg.pick_anchor(FRAME, DEFAULT)


got = drive(pick, [(drag, 255), ([], ENTER)])
check("drag then ENTER returns the dragged box",
      got and max(abs(a - b) for a, b in zip(got, want)) < 1e-9, str(got))
check("ESC cancels", drive(pick, [(drag, ESC)]) is None)
check("SPACE also accepts", drive(pick, [(drag, SPACE)]) is not None)

tiny = [(DOWN, 300, 300), (UP, 302, 301)]
got = drive(pick, [(tiny, 255), ([], ENTER), ([], ord("r")), ([], ENTER)])
check("stray click rejected, R restores default",
      got and max(abs(a - b) for a, b in zip(got, DEFAULT)) < 0.002, str(got))
check("no GUI falls back to the default anchor",
      drive(pick, [([], ENTER)], fail_window=True) == DEFAULT)

# ---------------------------------------------------------------------------
# Crop picker state machine
# ---------------------------------------------------------------------------
print("crop picker")
SCAN = np.random.randint(0, 255, (3715, 8256, 3), np.uint8)
Y0, Y1 = 1000, 3139
XR = [(238, 2182), (2182, 4154), (4154, 6120), (6120, 8061)]
S = min(1.0, 1400 / 8256, 900 / 3715)


def crop():
    return wg.pick_crop(SCAN, Y0, Y1, XR)


got = drive(crop, [([], ENTER)])
err = max([abs(got[0] - Y0), abs(got[1] - Y1)] +
          [abs(a - b) for (a, _), (b, _) in zip(got[2], XR)])
check("no-op accept round-trips the crop", err <= 3, f"max err {err}px")

top = round(Y0 * S)
d = [(DOWN, 700, top), (MOVE, 700, top - 30), (UP, 700, top - 30)]
got = drive(crop, [(d, 255), ([], ENTER)])
check("dragging the band top moves only that edge",
      abs(got[0] - round((top - 30) / S)) <= 1 and got[1] == Y1, f"band {got[0]}-{got[1]}")

cx = round(XR[1][0] * S)
d = [(DOWN, cx, 300), (MOVE, cx + 40, 300), (UP, cx + 40, 300)]
got = drive(crop, [(d, 255), ([], ENTER)])
check("dragging a cut moves it and keeps cuts ordered",
      abs(got[2][0][1] - round((cx + 40) / S)) <= 2
      and all(a[1] == b[0] for a, b in zip(got[2], got[2][1:])))

d = [(DOWN, 700, 400), (MOVE, 700, 200), (UP, 700, 200)]
got = drive(crop, [(d, 255), ([], ENTER)])
check("a click away from any line grabs nothing", (got[0], got[1]) == (Y0, Y1))

d = [(DOWN, 700, top), (MOVE, 700, 99999), (UP, 700, 99999)]
got = drive(crop, [(d, 255), ([], ENTER)])
check("band top cannot cross the bottom", got[0] < got[1], f"band {got[0]}-{got[1]}")

check("ESC cancels", drive(crop, [(d, ESC)]) is None)
got = drive(crop, [(d, 255), ([], ord("r")), ([], ENTER)])
check("R resets the crop", abs(got[0] - Y0) <= 1 and abs(got[1] - Y1) <= 1)
check("no GUI keeps the detected crop",
      drive(crop, [([], ENTER)], fail_window=True) == (Y0, Y1, XR))

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("all picker tests passed")
