#!/usr/bin/env python3
"""Local web UI for Wigglegramographer.

    python ui.py                       # open a scan from the page
    python ui.py samples/_DSC5466.jpg  # or start with one loaded

Serves a single page on 127.0.0.1. `wigglegram.py` stays the headless engine;
this is a client of it. Two stages:

  CROP    the whole scan, drag the band edges and the frame cuts
  WIGGLE  the looping animation, drag the subject anchor

Why the anchor preview is built on a precomputed parallax field rather than on
align_frames: ECC costs ~30s per anchor at full resolution, so exploring anchors
through it is impossible. Dense optical flow computed once (~0.4s) turns any
anchor's shift into a box average, which is instant. The preview is therefore
approximate; export re-runs the real ECC.
"""
import argparse
import base64
import http.server
import io
import json
import os
import re
import tempfile
import threading
import webbrowser

import numpy as np
import cv2

from wigglegram import (detect_band, detect_frames, crop_frames, normalize_sizes,
                        make_wigglegram, width_spread)

PREVIEW_H = 700          # working resolution for frames and flow
SCAN_W = 1400            # working resolution for the crop editor
STRIDE = 8               # coarse grid sent to the browser

STATE = {"path": None, "band": None, "cuts": None, "n_frames": 4, "inset": 0.01}
UPLOADS = os.path.join(tempfile.gettempdir(), "wigglegramographer")


# ----------------------------------------------------------------------------
# Preparation
# ----------------------------------------------------------------------------
def _coarse(a, stride=STRIDE):
    """Block-mean down to a coarse grid the browser can average cheaply."""
    h, w = a.shape[:2]
    gh, gw = h // stride, w // stride
    a = a[:gh * stride, :gw * stride]
    if a.ndim == 2:
        return a.reshape(gh, stride, gw, stride).mean(axis=(1, 3))
    return a.reshape(gh, stride, gw, stride, a.shape[2]).mean(axis=(1, 3))


def _jpeg(img, quality=88):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def detect_crop(force=False):
    """Fill in STATE band/cuts from the detectors if not already set."""
    gray = cv2.imread(STATE["path"], cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(STATE["path"])
    if force or not STATE["band"]:
        y0, y1 = detect_band(gray)
        STATE["band"] = [int(y0), int(y1)]
    y0, y1 = STATE["band"]
    if force or not STATE["cuts"]:
        xr, _ = detect_frames(gray[y0:y1, :], n_frames=STATE["n_frames"])
        STATE["cuts"] = [int(xr[0][0])] + [int(b) for _a, b in xr]
    return gray.shape


def scanview(force=False):
    """The crop stage: the whole scan plus the current band and cuts."""
    h, w = detect_crop(force)
    img = cv2.imread(STATE["path"], cv2.IMREAD_COLOR)
    scale = min(1.0, SCAN_W / w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return {"name": os.path.basename(STATE["path"]), "img": _jpeg(small, 82),
            "scanW": int(w), "scanH": int(h), "scale": scale,
            "band": STATE["band"], "cuts": STATE["cuts"],
            "frames": STATE["n_frames"]}


def prepare():
    """The wiggle stage: cropped frames, parallax field and lock-meter grids."""
    if not STATE["band"] or not STATE["cuts"]:
        detect_crop()            # never depend on scanview having run first
    img = cv2.imread(STATE["path"], cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(STATE["path"])
    y0, y1 = STATE["band"]
    cuts = STATE["cuts"]
    xr = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    frames = normalize_sizes(crop_frames(img, y0, y1, xr, inset=STATE["inset"]))

    scale = PREVIEW_H / frames[0].shape[0]
    small = [cv2.resize(f, (int(f.shape[1] * scale), PREVIEW_H),
                        interpolation=cv2.INTER_AREA) for f in frames]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in small]

    flows = [cv2.calcOpticalFlowFarneback(grays[0], g, None,
                                          0.5, 4, 25, 3, 7, 1.5, 0) for g in grays[1:]]

    # Structure tensor of the reference frame for the live lock meter. Sent as
    # three grids so the browser averages over a box and takes the smaller
    # eigenvalue itself -- eigenvalue-of-mean is not mean-of-eigenvalues.
    g0 = cv2.GaussianBlur(grays[0].astype(np.float32), (5, 5), 0)
    ix = cv2.Sobel(g0, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(g0, cv2.CV_32F, 0, 1, ksize=3)
    tensor = {k: _coarse(v).round(1).tolist()
              for k, v in (("xx", ix * ix), ("yy", iy * iy), ("xy", ix * iy))}

    coarse = [_coarse(f).round(2).tolist() for f in flows]
    return {"name": os.path.basename(STATE["path"]),
            "frames": [_jpeg(f) for f in small],
            "w": small[0].shape[1], "h": small[0].shape[0], "scale": scale,
            "grid": {"w": len(coarse[0][0]), "h": len(coarse[0])},
            "flows": coarse, "tensor": tensor,
            "band": STATE["band"], "cuts": STATE["cuts"],
            "spread": round(width_spread(xr), 2),
            "fullSize": [int(frames[0].shape[1]), int(frames[0].shape[0])]}


# ----------------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------------
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, fn):
        try:
            self._send(200, json.dumps(fn()))
        except Exception as exc:
            self._send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}))

    def do_GET(self):
        if self.path == "/":
            self._send(200, page(), "text/html; charset=utf-8")
        elif self.path.startswith("/api/scanview"):
            if not STATE["path"]:
                self._send(200, json.dumps({"empty": True}))
            else:
                self._json(lambda: scanview("force=1" in self.path))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b""

        if self.path == "/api/upload":
            def up():
                name = re.sub(r"[^A-Za-z0-9._-]", "_",
                              self.headers.get("X-Filename", "scan.jpg"))[-80:]
                os.makedirs(UPLOADS, exist_ok=True)
                dest = os.path.join(UPLOADS, name)
                with open(dest, "wb") as fh:
                    fh.write(raw)
                if cv2.imread(dest, cv2.IMREAD_GRAYSCALE) is None:
                    raise ValueError("not a readable image")
                STATE.update(path=dest, band=None, cuts=None)
                print(f"  loaded {name} ({len(raw) // 1024} KB)")
                return scanview()
            return self._json(up)

        req = json.loads(raw or b"{}")

        if self.path == "/api/open":
            def op():
                p = os.path.expanduser(req["path"].strip('"'))
                if cv2.imread(p, cv2.IMREAD_GRAYSCALE) is None:
                    raise ValueError(f"could not read {p}")
                STATE.update(path=p, band=None, cuts=None)
                return scanview()
            return self._json(op)

        if self.path == "/api/prepare":
            def pr():
                if req.get("band"):
                    STATE["band"] = [int(v) for v in req["band"]]
                if req.get("cuts"):
                    STATE["cuts"] = sorted(int(v) for v in req["cuts"])
                    STATE["n_frames"] = len(STATE["cuts"]) - 1
                return prepare()
            return self._json(pr)

        if self.path == "/api/export":
            def ex():
                out = req.get("out") or os.path.join(
                    "outputs", os.path.splitext(os.path.basename(STATE["path"]))[0]
                    + "_wiggle.gif")
                os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
                if not STATE["band"] or not STATE["cuts"]:
                    detect_crop()        # export must not depend on page-load order
                print(f"\nexporting {out} ...")
                res = make_wigglegram(
                    STATE["path"], out, n_frames=STATE["n_frames"],
                    fps=req.get("fps", 8), align=req.get("align", "translation"),
                    point=tuple(req["point"]) if req.get("point") else None,
                    anchor=tuple(req.get("anchor", (0.5, 0.6, 0.45, 0.55))),
                    band=tuple(STATE["band"]),
                    cuts=STATE["cuts"], repair=req.get("repair", False),
                    stabilize=float(req.get("stabilize", 1.0)),
                    nudge=[tuple(n) for n in req.get("nudge") or []] or None,
                    pingpong=req.get("pingpong", True), reverse=req.get("reverse", False),
                    max_height=req.get("maxHeight", 600), inset=STATE["inset"])
                res["out"] = os.path.abspath(out)
                res["ok"] = True
                return res
            return self._json(ex)

        self._send(404, json.dumps({"error": "not found"}))


PAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "ui_page.html")


def page():
    """Read the page off disk per request, so editing it only needs a refresh."""
    with io.open(PAGE_PATH, encoding="utf-8") as fh:
        return fh.read()


def main():
    ap = argparse.ArgumentParser(description="Local web UI for Wigglegramographer.")
    ap.add_argument("input", nargs="?", help="path to a scan (optional)")
    ap.add_argument("--port", type=int, default=8756)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--inset", type=float, default=0.01)
    ap.add_argument("--band", help="y0,y1 override")
    ap.add_argument("--cuts", help="n_frames+1 cut positions")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    STATE.update(path=args.input, n_frames=args.frames, inset=args.inset,
                 band=[int(v) for v in args.band.split(",")] if args.band else None,
                 cuts=[int(v) for v in args.cuts.split(",")] if args.cuts else None)

    url = f"http://127.0.0.1:{args.port}/"
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Wigglegramographer UI -> {url}")
    print(f"  scan: {args.input or '(open one from the page)'}\n  Ctrl+C to stop")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
