#!/usr/bin/env python3
"""
Wigglegramographer

Takes a sprocket scan of a 3D-camera negative (e.g. Nimslo 3D), auto-isolates the
central image band, splits it into the four frames at the dark inter-frame gaps,
optionally aligns them onto a subject anchor, and exports a looping ping-pong GIF.

Pipeline: ingest -> isolate band -> split into frames -> (align) -> sequence -> GIF.

Usage:
    python wigglegram.py INPUT [-o OUT.gif] [--frames 4] [--fps 8]
                         [--align translation|euclidean|none]
                         [--anchor cx,cy,w,h]   (fractions, 0-1; default 0.5,0.6,0.45,0.55)
                         [--pick-anchor] [--band y0,y1] [--cuts x0,...,xN]
                         [--repair-weak] [--no-pingpong] [--reverse]
                         [--max-height 600] [--inset 0.01] [--debug]

Examples:
    python wigglegram.py scan.jpg --debug
    python wigglegram.py scan.jpg --align none          # raw, unaligned wiggle
    python wigglegram.py scan.jpg --anchor 0.5,0.7,0.3,0.4   # tighter box, lower
    python wigglegram.py scan.jpg --pick-anchor         # fix crop, then pick subject
    python wigglegram.py scan.jpg --band 790,3140       # replay a picked crop
    python wigglegram.py scan.jpg --repair-weak         # infer frames that failed
    python wigglegram.py scan.jpg --reverse --fps 10
"""
import argparse
import os
import numpy as np
import cv2
from PIL import Image


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------
def detect_band(gray, rel_thr=0.35):
    """Vertical bounds of the image band, via a row-brightness profile.
    The band is the largest contiguous run of rows above an adaptive threshold."""
    row_mean = gray.mean(axis=1)
    thr = row_mean.min() + rel_thr * (row_mean.max() - row_mean.min())
    y0, y1 = _largest_run(row_mean > thr)
    if y0 is None:
        raise RuntimeError("Could not locate the image band (row profile too flat).")
    return y0, y1


def detect_frames(band_gray, n_frames=4, rel_thr=0.35, smooth=25, pct=90):
    """Split the band into n_frames at the dark inter-frame gaps.
    Returns [(x0, x1), ...] and the method used. Falls back to even division.

    Columns are profiled by a high percentile rather than a mean. A true
    inter-frame gap is dark for the *full* height of the band, so its percentile
    stays dark; dark image content (a doorway, a shadow) has bright pixels above
    or below it in the same column and so scores high. A mean cannot tell those
    apart and will happily cut straight through the middle of a frame.
    """
    w = band_gray.shape[1]
    col = np.percentile(band_gray, pct, axis=0)
    if smooth > 1:
        col = np.convolve(col, np.ones(smooth) / smooth, mode="same")

    thr = col.min() + rel_thr * (col.max() - col.min())
    runs = _all_runs(col < thr)

    edge = max(2, w // 100)
    interior = [(s, e) for (s, e) in runs if s > edge and e < w - 1 - edge]
    border_runs = [(s, e) for (s, e) in runs if s <= edge or e >= w - 1 - edge]
    left = max((e for (s, e) in border_runs if s <= edge), default=0)
    right = min((s for (s, e) in border_runs if e >= w - 1 - edge), default=w - 1)

    need = n_frames - 1
    if len(interior) >= need:
        interior.sort(key=lambda r: col[(r[0] + r[1]) // 2])   # darkest gaps first
        gaps = sorted(interior[:need], key=lambda r: r[0])
        cuts = [left] + [(s + e) // 2 for (s, e) in gaps] + [right]
        method = "gap-detected"
    else:
        cuts = list(np.linspace(left, right, n_frames + 1).astype(int))
        method = "even-split (fallback)"

    return [(cuts[i], cuts[i + 1]) for i in range(n_frames)], method


_SPREAD_WARN = 2.0         # percent; above this the split almost certainly missed
_MIN_CC = 0.80             # ECC correlation below this means the frame did not lock


def width_spread(xranges):
    """Coefficient of variation of the frame widths, in percent.

    The frames sit on a fixed mechanical pitch, so their widths should be
    near-identical. Correct splits measure well under 1%; splits that land off
    the gaps run several percent, which makes this a cheap self-check.
    """
    widths = np.array([x1 - x0 for (x0, x1) in xranges], dtype=float)
    return float(widths.std() / widths.mean() * 100)


def _largest_run(mask):
    best_len, best = 0, (None, None)
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i > best_len:
                best_len, best = j - i, (i, j - 1)
            i = j
        else:
            i += 1
    return best


def _all_runs(mask):
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


# ----------------------------------------------------------------------------
# Cropping
# ----------------------------------------------------------------------------
def crop_frames(img, y0, y1, xranges, inset=0.01):
    """Crop each frame, shaving a small inset off every edge to drop residual rebate."""
    frames = []
    for (x0, x1) in xranges:
        dx, dy = int((x1 - x0) * inset), int((y1 - y0) * inset)
        frames.append(img[y0 + dy:y1 - dy, x0 + dx:x1 - dx])
    return frames


def normalize_sizes(frames):
    """Resize every frame to a common size so the sequence registers."""
    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    return [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames]


# ----------------------------------------------------------------------------
# Alignment (Phase 2)
# ----------------------------------------------------------------------------
def _anchor_mask(H, W, anchor):
    cx, cy, bw, bh = anchor
    ax0, ax1 = int((cx - bw / 2) * W), int((cx + bw / 2) * W)
    ay0, ay1 = int((cy - bh / 2) * H), int((cy + bh / 2) * H)
    # Clamp before returning: the box is also used to slice the phaseCorrelate
    # fallback, where a negative bound would silently take the wrong region.
    ax0, ax1 = max(0, ax0), min(W, ax1)
    ay0, ay1 = max(0, ay0), min(H, ay1)
    m = np.zeros((H, W), np.uint8)
    m[ay0:ay1, ax0:ax1] = 255
    return m, (ax0, ay0, ax1, ay1)


def align_frames(frames, anchor=(0.5, 0.6, 0.45, 0.55), mode="translation",
                 ref_index=0, repair=False):
    """Register frames so the subject inside the anchor box stays fixed.

    Uses masked ECC so only the subject region drives the estimate (the busy
    background is ignored). For translation mode, the sequence is recentred on the
    mean position to minimise edge cropping.

    Returns (frames_cropped, box, shifts, ccs). `ccs` is the per-frame ECC
    correlation score: ~0.85+ is a solid lock, below ~0.75 means that frame's
    shift is not trustworthy. Callers should surface it — a weak lock is the
    difference between a good wiggle and one frame flying off on its own.
    """
    H, W = frames[0].shape[:2]
    mask, box = _anchor_mask(H, W, anchor)
    ref = cv2.cvtColor(frames[ref_index], cv2.COLOR_BGR2GRAY).astype(np.float32)
    mt = cv2.MOTION_TRANSLATION if mode == "translation" else cv2.MOTION_EUCLIDEAN
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-7)

    warps, ccs = [], []
    for i, f in enumerate(frames):
        warp = np.eye(2, 3, dtype=np.float32)
        cc = 1.0
        if i != ref_index:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            try:
                cc, warp = cv2.findTransformECC(ref, g, warp, mt, crit, mask, 5)
            except cv2.error:
                # Fallback: phase correlation on the anchor crop (translation only).
                # The Hann window is load-bearing: without it, spectral leakage from
                # the sharp crop edges parks the peak at zero, so the fallback
                # silently reports "no shift" for every frame.
                cc = 0.0
                x0, y0, x1, y1 = box
                rc = np.ascontiguousarray(ref[y0:y1, x0:x1])
                gc = np.ascontiguousarray(g[y0:y1, x0:x1])
                win = cv2.createHanningWindow((rc.shape[1], rc.shape[0]), cv2.CV_32F)
                (dx, dy), _ = cv2.phaseCorrelate(rc, gc, win)
                warp[0, 2], warp[1, 2] = dx, dy
        warps.append(warp)
        ccs.append(float(cc))

    if repair:
        # The four lenses sit on a fixed horizontal pitch, so for any one depth
        # plane the shift is linear in frame index. Fit that line through the
        # frames that locked well and use it to replace the ones that did not.
        # Needs two good frames to define a line; with fewer, leave it alone.
        good = [i for i, c in enumerate(ccs) if c >= _MIN_CC]
        weak = [i for i, c in enumerate(ccs) if c < _MIN_CC]
        if weak and len(good) >= 2:
            for axis in (0, 1):
                fit = np.polyfit(good, [warps[i][axis, 2] for i in good], 1)
                for i in weak:
                    warps[i][axis, 2] = float(np.polyval(fit, i))

    if mode == "translation":  # recentre on mean translation
        mean_t = np.mean([[w[0, 2], w[1, 2]] for w in warps], axis=0)
        for w in warps:
            w[0, 2] -= mean_t[0]
            w[1, 2] -= mean_t[1]

    aligned, valids, shifts = [], [], []
    for f, w in zip(frames, warps):
        aligned.append(cv2.warpAffine(f, w, (W, H),
                       flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP))
        valids.append(cv2.warpAffine(np.full((H, W), 255, np.uint8), w, (W, H),
                      flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP))
        shifts.append((float(w[0, 2]), float(w[1, 2])))

    allv = np.min(np.stack(valids), axis=0)
    ys, xs = np.where(allv > 0)
    cy0, cy1, cx0, cx1 = ys.min(), ys.max(), xs.min(), xs.max()
    return [f[cy0:cy1, cx0:cx1] for f in aligned], box, shifts, ccs


# ----------------------------------------------------------------------------
# Interactive anchor picking (Phase 3)
# ----------------------------------------------------------------------------
_WIN = "Wigglegramographer - drag a box over the subject"
_WIN_CROP = "Wigglegramographer - drag the band edges and the frame cuts"
_MIN_AREA = 0.015          # reject stray clicks: box must cover >=1.5% of the frame
_GRAB_PX = 12              # how close a click must be to grab a line, in preview px


def _close(win):
    cv2.destroyWindow(win)
    cv2.waitKey(1)


def _preview(img, max_w=1400, max_h=900):
    """Downscale for on-screen use. Returns (preview, scale)."""
    H, W = img.shape[:2]
    s = min(1.0, max_w / W, max_h / H)
    disp = cv2.resize(img, (max(1, round(W * s)), max(1, round(H * s))),
                      interpolation=cv2.INTER_AREA)
    return disp, s


def pick_crop(img, y0, y1, xranges):
    """Drag the band's top/bottom edges and the inter-frame cuts on the full scan.

    This is the step that recovers over-cropping: `detect_band` clips the top and
    bottom of dark scenes, and nothing downstream can put that back.

    Returns (y0, y1, xranges) in full-resolution pixels, None if cancelled, or the
    inputs unchanged when no GUI is available. Unlike the anchor box these are
    absolute coordinates, so the preview scale does not cancel and is divided out.
    """
    disp, s = _preview(img)
    DH, DW = disp.shape[:2]
    start = [xranges[0][0]] + [b for (_a, b) in xranges]
    st = {"top": y0 * s, "bot": y1 * s, "cuts": [c * s for c in start], "grab": None}

    def reset():
        st["top"], st["bot"] = y0 * s, y1 * s
        st["cuts"] = [c * s for c in start]

    def nearest(x, y):
        cands = [("top", abs(y - st["top"])), ("bot", abs(y - st["bot"]))]
        cands += [(("cut", i), abs(x - c)) for i, c in enumerate(st["cuts"])]
        name, dist = min(cands, key=lambda t: t[1])
        return name if dist <= _GRAB_PX else None

    def on_mouse(event, x, y, flags, _):
        x, y = min(DW, max(0, x)), min(DH, max(0, y))
        if event == cv2.EVENT_LBUTTONDOWN:
            st["grab"] = nearest(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            st["grab"] = None
        elif event == cv2.EVENT_MOUSEMOVE and st["grab"] is not None:
            g = st["grab"]
            if g == "top":
                st["top"] = min(y, st["bot"] - 10)
            elif g == "bot":
                st["bot"] = max(y, st["top"] + 10)
            else:
                st["cuts"][g[1]] = x

    def result():
        cuts = sorted(int(round(c / s)) for c in st["cuts"])
        return (int(round(st["top"] / s)), int(round(st["bot"] / s)),
                [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)])

    try:
        cv2.namedWindow(_WIN_CROP, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(_WIN_CROP, on_mouse)
    except cv2.error:
        print("  crop      : no GUI available, keeping the detected crop")
        return y0, y1, xranges

    print("  crop      : drag the green band edges / red cuts "
          "(ENTER accept, R reset, ESC cancel)")
    while True:
        canvas = (disp * 0.4).astype(np.uint8)
        t, b = int(st["top"]), int(st["bot"])
        canvas[max(0, t):max(0, b)] = disp[max(0, t):max(0, b)]
        for yy in (t, b):
            cv2.line(canvas, (0, yy), (DW, yy), (0, 255, 0), 2)
        for c in st["cuts"]:
            cv2.line(canvas, (int(c), t), (int(c), b), (0, 0, 255), 2)
        _, _, xr = result()
        cv2.putText(canvas, "drag edges/cuts   ENTER: accept   R: reset   ESC: cancel",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "band %d-%d   width spread %.1f%%"
                    % (int(round(st["top"] / s)), int(round(st["bot"] / s)), width_spread(xr)),
                    (10, DH - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        try:
            cv2.imshow(_WIN_CROP, canvas)
            key = cv2.waitKey(20) & 0xFF
            if cv2.getWindowProperty(_WIN_CROP, cv2.WND_PROP_VISIBLE) < 1:
                key = 27
        except cv2.error:
            print("  crop      : no GUI available, keeping the detected crop")
            return y0, y1, xranges

        if key == 27:
            _close(_WIN_CROP)
            return None
        if key in (13, 32):
            _close(_WIN_CROP)
            return result()
        if key in (ord("r"), ord("R")):
            reset()


def _anchor_to_box(anchor, DW, DH):
    """(cx, cy, w, h) fractions -> display-space pixel box."""
    cx, cy, bw, bh = anchor
    return [int((cx - bw / 2) * DW), int((cy - bh / 2) * DH),
            int((cx + bw / 2) * DW), int((cy + bh / 2) * DH)]


def _box_to_anchor(box, DW, DH):
    """Display-space pixel box -> (cx, cy, w, h) fractions.

    Dividing by the *displayed* size means the preview scale cancels out, so
    there is nothing to un-scale and no rounding path back to full resolution.
    Edges are clamped to the frame first (clamping is monotonic, so the sorted
    order holds), which keeps the result a real in-frame box rather than four
    independently-clipped numbers.
    """
    x0, x1 = sorted((box[0], box[2]))
    y0, y1 = sorted((box[1], box[3]))
    x0, x1 = (min(DW, max(0, v)) for v in (x0, x1))
    y0, y1 = (min(DH, max(0, v)) for v in (y0, y1))
    return ((x0 + x1) / 2 / DW, (y0 + y1) / 2 / DH, (x1 - x0) / DW, (y1 - y0) / DH)


def pick_anchor(frame, default_anchor, max_w=1400, max_h=900):
    """Let the user drag the subject anchor box on the reference frame.

    Shows a downscaled copy with the current anchor pre-seeded. Returns
    (cx, cy, w, h) fractions, None if the user cancelled, or default_anchor
    unchanged when no GUI is available (so scripted runs never hard-fail).
    """
    disp, _s = _preview(frame, max_w, max_h)
    DH, DW = disp.shape[:2]
    st = {"box": _anchor_to_box(default_anchor, DW, DH), "drag": False}

    def on_mouse(event, x, y, flags, _):
        # Clamp to DW/DH, not DW-1: these are exclusive box edges (and slice
        # bounds), so a drag off the right edge should reach a full 1.0.
        x, y = min(DW, max(0, x)), min(DH, max(0, y))
        if event == cv2.EVENT_LBUTTONDOWN:
            st["drag"], st["box"] = True, [x, y, x, y]
        elif event == cv2.EVENT_MOUSEMOVE and st["drag"]:
            st["box"][2], st["box"][3] = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            st["drag"] = False
            st["box"][2], st["box"][3] = x, y

    def _no_gui():
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print("  anchor    : no GUI available, falling back to the default box")
        return default_anchor

    try:
        cv2.namedWindow(_WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(_WIN, on_mouse)
    except cv2.error:
        return _no_gui()

    print("  anchor    : drag a box over the subject "
          "(ENTER accept, R reset, ESC cancel)")
    while True:
        canvas = (disp * 0.35).astype(np.uint8)
        x0, x1 = sorted((st["box"][0], st["box"][2]))
        y0, y1 = sorted((st["box"][1], st["box"][3]))
        canvas[y0:y1, x0:x1] = disp[y0:y1, x0:x1]      # undim the selection
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 2)
        anchor = _box_to_anchor(st["box"], DW, DH)
        cv2.putText(canvas, "drag: subject   ENTER: accept   R: reset   ESC: cancel",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "--anchor %.3f,%.3f,%.3f,%.3f" % anchor,
                    (10, DH - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        try:
            cv2.imshow(_WIN, canvas)
            key = cv2.waitKey(20) & 0xFF
            if cv2.getWindowProperty(_WIN, cv2.WND_PROP_VISIBLE) < 1:
                key = 27                                # window closed == cancel
        except cv2.error:
            return _no_gui()

        if key == 27:                                   # ESC
            _close(_WIN)
            return None
        if key in (13, 32):                             # ENTER / SPACE
            if anchor[2] * anchor[3] < _MIN_AREA:
                print("  anchor    : that box is too small - drag a larger one")
                continue
            _close(_WIN)
            return anchor
        if key in (ord("r"), ord("R")):
            st["box"] = _anchor_to_box(default_anchor, DW, DH)


# ----------------------------------------------------------------------------
# Sequencing & export
# ----------------------------------------------------------------------------
def build_sequence(frames, pingpong=True, reverse=False):
    seq = list(frames[::-1] if reverse else frames)
    if pingpong and len(seq) > 2:
        seq = seq + seq[-2:0:-1]            # 1,2,3,4,3,2 -> seamless loop
    return seq


def export_gif(seq, path, fps=8, max_height=600):
    """Write a looping GIF with explicit per-frame timing and disposal.

    GIF stores the frame delay in centiseconds inside a Graphic Control
    Extension. Quantise to centiseconds here rather than leaving it to the
    encoder: a delay that rounds to 0 makes some decoders (notably Android's)
    fall back to a default speed or refuse to animate, and a delay under 2cs is
    widely clamped. Disposal is set explicitly to "do not dispose" — the frames
    are full-size and opaque, so each simply replaces the last.

    Returns the effective fps, which the centisecond grid may round.
    """
    frames = []
    for f in seq:
        if max_height and f.shape[0] > max_height:
            s = max_height / f.shape[0]
            f = cv2.resize(f, (int(f.shape[1] * s), max_height), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))

    cs = max(2, int(round(100.0 / fps)))
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=256) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=cs * 10,
                loop=0, disposal=1, optimize=False)
    return 100.0 / cs


# ----------------------------------------------------------------------------
# Debug
# ----------------------------------------------------------------------------
def save_debug(img, y0, y1, xranges, frames, box, base):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (img.shape[1] - 1, y1), (0, 255, 0), 4)
    for (x0, x1) in xranges:
        cv2.line(overlay, (x0, y0), (x0, y1), (0, 0, 255), 3)
        cv2.line(overlay, (x1, y0), (x1, y1), (0, 0, 255), 3)
    cv2.imwrite(f"{base}_debug_detect.png", overlay)

    norm = normalize_sizes(frames)
    if box is not None:
        bx0, by0, bx1, by1 = box
        norm = [f.copy() for f in norm]
        cv2.rectangle(norm[0], (bx0, by0), (bx1, by1), (0, 255, 0), 3)
    contact = np.hstack([cv2.copyMakeBorder(f, 0, 0, 8, 8, cv2.BORDER_CONSTANT,
                                            value=(255, 255, 255)) for f in norm])
    cv2.imwrite(f"{base}_debug_frames.png", contact)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def make_wigglegram(path, out, n_frames=4, fps=8, align="translation",
                    anchor=(0.5, 0.6, 0.45, 0.55), pick=False, band=None, cuts=None,
                    repair=False, pingpong=True, reverse=False, max_height=600,
                    inset=0.01, debug=False):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    y0, y1 = detect_band(gray)
    xranges, method = detect_frames(gray[y0:y1, :], n_frames=n_frames)
    if band is not None:
        y0, y1 = band
        method += ", manual band"
    if cuts is not None:
        if len(cuts) != n_frames + 1:
            raise SystemExit(f"--cuts needs {n_frames + 1} values for {n_frames} frames")
        xranges = [(cuts[i], cuts[i + 1]) for i in range(n_frames)]
        method += ", manual cuts"

    if pick:
        picked = pick_crop(img, y0, y1, xranges)
        if picked is None:
            raise SystemExit("  crop      : cancelled - nothing written")
        y0, y1, xranges = picked
        print(f"  crop      : --band {y0},{y1} "
              f"--cuts {','.join(str(x0) for x0, _ in xranges)},{xranges[-1][1]}")

    spread = width_spread(xranges)
    print(f"  band rows : {y0}-{y1}  ({y1 - y0}px tall)")
    print(f"  frames    : {method}, width spread {spread:.1f}%")
    if spread > _SPREAD_WARN:
        print(f"  WARNING   : uneven frame widths ({spread:.1f}%) - the split likely missed")
        print( "              the gaps, so frames may straddle two images. Re-run with")
        print( "              --debug and check the red cut lines in *_debug_detect.png.")

    frames = normalize_sizes(crop_frames(img, y0, y1, xranges, inset=inset))

    box = None
    if align != "none":
        if pick:
            picked = pick_anchor(frames[0], anchor)
            if picked is None:
                raise SystemExit("  anchor    : cancelled - nothing written")
            anchor = picked
            print("  anchor    : --anchor %.3f,%.3f,%.3f,%.3f" % anchor)
        frames, box, shifts, ccs = align_frames(frames, anchor=anchor, mode=align,
                                                repair=repair)
        n_good = sum(1 for cc in ccs if cc >= _MIN_CC)
        fixed = repair and n_good >= 2
        print(f"  align     : {align}, anchor={anchor}")
        for i, ((dx, dy), cc) in enumerate(zip(shifts, ccs), 1):
            if cc >= _MIN_CC:
                flag = ""
            else:
                flag = "   <- weak, repaired" if fixed else "   <- weak lock"
            print(f"    frame {i}: dx={dx:+7.2f}  dy={dy:+7.2f}  cc={cc:.3f}{flag}")
        weak = [i for i, cc in enumerate(ccs, 1) if cc < _MIN_CC]
        if weak and not fixed:
            print(f"  WARNING   : frame(s) {weak} did not lock onto the anchor "
                  f"(cc < {_MIN_CC}).")
            if repair:
                print( "              Too few good frames to fit a repair line, so their")
                print( "              measured shifts were kept as-is.")
            else:
                print( "              Those frames will jump. Re-run with --pick-anchor and")
                print( "              choose a smaller, high-contrast subject detail,")
                print( "              or pass --repair-weak to infer them from the others.")
        frames = normalize_sizes(frames)

    seq = build_sequence(frames, pingpong=pingpong, reverse=reverse)
    eff = export_gif(seq, out, fps=fps, max_height=max_height)
    rounded = "" if abs(eff - fps) < 0.05 else f" (requested {fps}, rounded to the GIF 10ms grid)"
    print(f"  wrote     : {out}  ({len(seq)} frames @ {eff:.2f}fps{rounded}, "
          f"{frames[0].shape[1]}x{frames[0].shape[0]})")

    if debug:
        base = os.path.splitext(out)[0]
        save_debug(img, y0, y1, xranges, frames, box, base)
        print(f"  debug     : {base}_debug_detect.png, {base}_debug_frames.png")


def _parse_anchor(s):
    v = [float(x) for x in s.split(",")]
    if len(v) != 4:
        raise argparse.ArgumentTypeError("anchor must be cx,cy,w,h (four fractions)")
    return tuple(v)


def _parse_band(s):
    v = [int(x) for x in s.split(",")]
    if len(v) != 2 or v[0] >= v[1]:
        raise argparse.ArgumentTypeError("band must be y0,y1 pixels with y0 < y1")
    return tuple(v)


def _parse_cuts(s):
    v = sorted(int(x) for x in s.split(","))
    if len(v) < 3:
        raise argparse.ArgumentTypeError("cuts must be n_frames+1 pixel positions")
    return v


def main():
    ap = argparse.ArgumentParser(description="Turn a 3D-camera negative scan into a wigglegram GIF.")
    ap.add_argument("input", help="path to the scan (jpg/png/webp/tif)")
    ap.add_argument("-o", "--output", help="output GIF path (default: <input>_wiggle.gif)")
    ap.add_argument("--frames", type=int, default=4, help="number of frames on the scan (default 4)")
    ap.add_argument("--fps", type=float, default=8, help="playback speed (default 8)")
    ap.add_argument("--align", choices=["translation", "euclidean", "none"], default="translation",
                    help="subject alignment mode (default translation)")
    ap.add_argument("--anchor", type=_parse_anchor, default=(0.5, 0.6, 0.45, 0.55),
                    help="subject box as cx,cy,w,h fractions (default 0.5,0.6,0.45,0.55)")
    ap.add_argument("--pick-anchor", action="store_true",
                    help="interactive: adjust the crop, then drag the subject box")
    ap.add_argument("--band", type=_parse_band,
                    help="override the band as y0,y1 in pixels")
    ap.add_argument("--cuts", type=_parse_cuts,
                    help="override the frame cuts as n_frames+1 pixel positions")
    ap.add_argument("--repair-weak", action="store_true",
                    help="replace shifts of frames that failed to lock with a "
                         "linear-parallax prediction from the frames that did")
    ap.add_argument("--no-pingpong", action="store_true", help="loop 1234 instead of 123432")
    ap.add_argument("--reverse", action="store_true", help="reverse frame order (flip wiggle direction)")
    ap.add_argument("--max-height", type=int, default=600, help="cap output height in px (default 600)")
    ap.add_argument("--inset", type=float, default=0.01, help="edge trim per frame, fraction (default 0.01)")
    ap.add_argument("--debug", action="store_true", help="also write detection + contact-sheet images")
    args = ap.parse_args()

    out = args.output or (os.path.splitext(args.input)[0] + "_wiggle.gif")
    print(f"Wigglegramographer -> {args.input}")
    make_wigglegram(args.input, out, n_frames=args.frames, fps=args.fps, align=args.align,
                    anchor=args.anchor, pick=args.pick_anchor, band=args.band,
                    cuts=args.cuts, repair=args.repair_weak,
                    pingpong=not args.no_pingpong, reverse=args.reverse,
                    max_height=args.max_height, inset=args.inset, debug=args.debug)


if __name__ == "__main__":
    main()
