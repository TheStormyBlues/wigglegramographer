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
            self._send(200, PAGE, "text/html; charset=utf-8")
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
                    pingpong=req.get("pingpong", True), reverse=req.get("reverse", False),
                    max_height=req.get("maxHeight", 600), inset=STATE["inset"])
                res["out"] = os.path.abspath(out)
                res["ok"] = True
                return res
            return self._json(ex)

        self._send(404, json.dumps({"error": "not found"}))


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Wigglegramographer</title>
<style>
:root{--bg:#16181d;--panel:#1e2128;--line:#2e323c;--fg:#e6e8ec;--dim:#8b919e;
      --accent:#6ea8fe;--good:#5fd08a;--warn:#e8b750;--bad:#e06c6c}
*{box-sizing:border-box}
body{margin:0;font:14px/1.45 ui-sans-serif,system-ui,Segoe UI,sans-serif;
     background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:12px;padding:10px 16px;
       border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:14px;font-weight:600;margin:0}
.file{color:var(--dim);font-size:13px;max-width:230px;overflow:hidden;
      text-overflow:ellipsis;white-space:nowrap}
.sp{flex:1}
.tabs{display:flex;gap:2px;background:#2a2e37;border-radius:6px;padding:2px}
.tabs button{background:none;border:0;padding:5px 14px;border-radius:4px;
             color:var(--dim);cursor:pointer;font-size:13px}
.tabs button.on{background:var(--accent);color:#10131a;font-weight:600}
main{flex:1;display:flex;min-height:0}
#stage{flex:1;display:flex;flex-direction:column;align-items:center;
       justify-content:center;padding:18px;gap:12px;min-width:0}
#wrap{width:fit-content;position:relative}
/* One canvas per stage: the overlays are drawn onto the same surface as the
   image. A separate overlay element has to track a canvas that rescales, and
   any mismatch shows up as dead space around the picture. */
canvas{display:block;border-radius:4px;cursor:crosshair;background:#000;
       max-height:64vh;max-width:100%;width:auto;height:auto}
#strip{display:flex;gap:8px}
#strip .f{text-align:center;color:var(--dim);font-size:11px}
#strip img{height:60px;border-radius:3px;display:block;border:2px solid transparent}
#strip .on img{border-color:var(--accent)}
#strip .cc{font-variant-numeric:tabular-nums;font-size:10px;color:var(--dim)}
#strip .cc.good{color:var(--good)}
#strip .cc.bad{color:var(--bad)}
#result{font-size:12px;line-height:1.5}
#result.bad{color:var(--warn)}
aside{width:320px;border-left:1px solid var(--line);background:var(--panel);
      padding:14px;overflow:auto;display:flex;flex-direction:column;gap:16px}
.sec h2{font-size:11px;text-transform:uppercase;letter-spacing:.09em;
        color:var(--dim);margin:0 0 9px}
.row{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:7px 0}
.row label{color:var(--dim)}
.val{font-variant-numeric:tabular-nums}
input[type=range]{width:145px;accent-color:var(--accent)}
input[type=text]{background:#161920;border:1px solid var(--line);color:var(--fg);
                 border-radius:5px;padding:6px 8px;width:100%;font-size:13px}
.meter{height:7px;border-radius:4px;background:#2a2e37;overflow:hidden}
.meter i{display:block;height:100%;background:var(--good);transition:width .1s}
button{background:#2b303a;color:var(--fg);border:1px solid var(--line);
       border-radius:5px;padding:7px 11px;cursor:pointer;font-size:13px}
button:hover{border-color:var(--accent)}
button.pri{background:var(--accent);color:#10131a;border-color:var(--accent);font-weight:600}
button:disabled{opacity:.45;cursor:default}
.hint{color:var(--dim);font-size:12px;line-height:1.5}
.tag{padding:1px 8px;border-radius:99px;font-size:11px;background:#2a2e37}
.tag.good{background:rgba(95,208,138,.16);color:var(--good)}
.tag.warn{background:rgba(232,183,80,.16);color:var(--warn)}
.tag.bad{background:rgba(224,108,108,.16);color:var(--bad)}
#msg{color:var(--dim);font-size:12px}
.empty{text-align:center;color:var(--dim);max-width:420px}
.empty h3{color:var(--fg);font-size:16px;margin:0 0 6px}
.hide{display:none!important}
</style></head><body>
<header>
  <h1>Wigglegramographer</h1><span class="file" id="fname">no scan</span>
  <label class="btn"><button id="openBtn">Open…</button>
    <input type="file" id="file" accept="image/*" class="hide"></label>
  <div class="tabs"><button id="tabCrop">Crop</button>
    <button id="tabWig" class="on">Wiggle</button></div>
  <span class="sp"></span><span id="msg"></span>
  <button id="export" class="pri" disabled>Export GIF</button>
</header>
<main>
  <div id="stage">
    <div id="empty" class="empty"><h3>Open a scan to start</h3>
      <p>A flatbed sprocket scan of a four-frame 3D negative. You can also pass
         one on the command line.</p>
      <input type="text" id="pathIn" placeholder="…or paste a path and press Enter">
    </div>
    <div id="wrap" class="hide"><canvas id="cv"></canvas></div>
    <div id="strip"></div>
  </div>
  <aside>
    <div class="sec" id="cropSec">
      <h2>Crop</h2>
      <div class="row"><label>band</label><span class="val" id="band">—</span></div>
      <div class="row"><label>height</label><span class="val" id="bandh">—</span></div>
      <div class="row"><label>frame widths</label><span id="spread" class="tag">—</span></div>
      <div class="row"><button id="apply" class="pri">Apply crop</button>
        <button id="redetect">Re-detect</button></div>
      <div class="hint" id="cropHint">Switch to the Crop tab to drag the green band
        edges and the red frame cuts. Even frame widths mean the cuts found the
        gaps.</div>
    </div>
    <div class="sec" id="anchorSec">
      <h2>Anchor</h2>
      <div class="row"><label>lock confidence</label><span class="val" id="lockv">—</span></div>
      <div class="meter"><i id="lock" style="width:0%"></i></div>
      <div class="row"><label>region</label><span class="val" id="regv">—</span></div>
      <div class="row"><label>parallax</label><span class="val" id="plx">—</span></div>
      <div class="row"><button id="dep">Depth overlay</button>
        <button id="reg">Anchor mask</button></div>
      <div class="hint">Click the thing you want to hold still. The region around it
        is grown to match that point's distance from the camera, so it follows the
        subject instead of straddling foreground and background. <b>M</b> toggles the
        mask.</div>
    </div>
    <div class="sec">
      <h2>Playback</h2>
      <div class="row"><label>fps</label>
        <input type="range" id="fps" min="2" max="24" step="1" value="8">
        <span class="val" id="fpsv">8</span></div>
      <div class="row"><label>ping-pong</label><input type="checkbox" id="pp" checked></div>
      <div class="row"><label>reverse</label><input type="checkbox" id="rev"></div>
    </div>
    <div class="sec">
      <h2>Export</h2>
      <div class="row"><label>height</label>
        <input type="range" id="mh" min="300" max="1400" step="50" value="600">
        <span class="val" id="mhv">600</span></div>
      <div class="row"><label>repair weak frames</label><input type="checkbox" id="rp"></div>
      <div class="hint">Export re-runs the real ECC alignment at full resolution, so
        it takes a while and can differ slightly from this preview.</div>
      <div id="result"></div>
    </div>
  </aside>
</main>
<script>
const $ = s => document.querySelector(s);
let D = null, S = null, imgs = [], scanImg = null, mode = 'wiggle';
let point = {x:.5, y:.6}, showDepth = false, showRegion = true, drag = null;
let shifts = [], padX = 0, padY = 0, depthCanvas = null, regionCanvas = null, region = null;
let t0 = 0, step = 0, cur = 0;
const msg = t => $('#msg').textContent = t || '';

// --- loading ---------------------------------------------------------------
async function api(url, opts) {
  const r = await fetch(url, opts); const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}
async function boot() {
  const j = await api('/api/scanview');
  if (j.empty) { setMode('wiggle'); return; }
  gotScan(j); await doPrepare();
}
function gotScan(j) {
  S = j; $('#fname').textContent = j.name;
  scanImg = new Image(); scanImg.onload = () => { if (mode === 'crop') render(); };
  scanImg.src = j.img;
  $('#empty').classList.add('hide'); $('#wrap').classList.remove('hide');
  syncCropPanel();
}
async function doPrepare(body) {
  msg('computing parallax field…'); $('#export').disabled = true;
  try {
    D = await api('/api/prepare', {method:'POST', body: JSON.stringify(body || {})});
    imgs = await Promise.all(D.frames.map(src => new Promise((res, rej) => {
      const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; })));
    $('#strip').innerHTML = imgs.map((im, k) =>
      `<div class="f" id="f${k}"><img src="${im.src}"><div>${k+1}</div>` +
      `<div class="cc" id="cc${k}">&nbsp;</div></div>`).join('');
    S.band = D.band; S.cuts = D.cuts; syncCropPanel();
    buildDepth(); update(); setMode('wiggle'); msg('');
    $('#export').disabled = false;
  } catch (e) { msg(e.message); }
}
$('#openBtn').onclick = () => $('#file').click();
$('#file').onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  msg('uploading…');
  try {
    gotScan(await api('/api/upload', {method:'POST', body: f,
      headers: {'X-Filename': f.name}}));
    await doPrepare();
  } catch (err) { msg(err.message); }
};
$('#pathIn').onkeydown = async e => {
  if (e.key !== 'Enter' || !e.target.value.trim()) return;
  msg('opening…');
  try {
    gotScan(await api('/api/open', {method:'POST',
      body: JSON.stringify({path: e.target.value})}));
    await doPrepare();
  } catch (err) { msg(err.message); }
};
$('#redetect').onclick = async () => {
  msg('re-detecting…');
  try { gotScan(await api('/api/scanview?force=1')); await doPrepare(); }
  catch (e) { msg(e.message); }
};
$('#apply').onclick = () => doPrepare({band: S.band, cuts: S.cuts});

// --- stage switching -------------------------------------------------------
function setMode(m) {
  mode = m;
  $('#tabCrop').classList.toggle('on', m === 'crop');
  $('#tabWig').classList.toggle('on', m === 'wiggle');
  $('#strip').classList.toggle('hide', m === 'crop');
  $('#anchorSec').classList.toggle('hide', m === 'crop');
  $('#cropHint').textContent = m === 'crop'
    ? 'Drag the green band edges or a red cut line, then Apply crop.'
    : 'Switch to the Crop tab to adjust the band and frame cuts.';
  render();
}
$('#tabCrop').onclick = () => S && setMode('crop');
$('#tabWig').onclick = () => D && setMode('wiggle');

function syncCropPanel() {
  if (!S) return;
  $('#band').textContent = S.band[0] + '–' + S.band[1];
  $('#bandh').textContent = (S.band[1] - S.band[0]) + 'px';
  const w = []; for (let i = 1; i < S.cuts.length; i++) w.push(S.cuts[i] - S.cuts[i-1]);
  const m = w.reduce((a,b)=>a+b,0)/w.length;
  const sd = Math.sqrt(w.reduce((a,b)=>a+(b-m)*(b-m),0)/w.length) / m * 100;
  const t = $('#spread'); t.textContent = sd.toFixed(1) + '%';
  t.className = 'tag ' + (sd < 1 ? 'good' : sd < 2 ? 'warn' : 'bad');
}

// --- depth-consistent region around the clicked point ----------------------
// Mirrors region_from_point() in the engine: parallax is a proxy for distance,
// so growing by flow similarity gives a region that follows the subject and
// stops at its silhouette. A rectangle would straddle two depth planes.
const TOL = [0.3, 0.45, 0.65, 0.9, 1.3, 1.8, 2.5, 3.5, 5, 7];
const R_MIN = 0.02, R_MAX = 0.35;

function flood(dist, sx, sy, tol) {
  const g = D.grid, seen = new Uint8Array(g.w * g.h), stack = [sy * g.w + sx];
  if (dist[sy][sx] >= tol) return {mask: seen, count: 0};
  seen[sy * g.w + sx] = 1;
  let count = 1;
  while (stack.length) {
    const i = stack.pop(), x = i % g.w, y = (i - x) / g.w;
    for (const [dx, dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= g.w || ny >= g.h) continue;
      const j = ny * g.w + nx;
      if (seen[j] || dist[ny][nx] >= tol) continue;
      seen[j] = 1; count++; stack.push(j);
    }
  }
  return {mask: seen, count};
}

function growRegion() {
  const g = D.grid;
  const sx = Math.min(g.w - 1, Math.max(0, Math.floor(point.x * g.w)));
  const sy = Math.min(g.h - 1, Math.max(0, Math.floor(point.y * g.h)));
  const seed = D.flows.map(fl => fl[sy][sx]);
  const dist = [];
  for (let y = 0; y < g.h; y++) {
    const row = new Float32Array(g.w);
    for (let x = 0; x < g.w; x++) {
      let m = 0;
      for (let k = 0; k < D.flows.length; k++) {
        const a = D.flows[k][y][x][0] - seed[k][0], b = D.flows[k][y][x][1] - seed[k][1];
        m = Math.max(m, Math.hypot(a, b));
      }
      row[x] = m;
    }
    dist.push(row);
  }
  const total = g.w * g.h;
  let best = null;
  for (const tol of TOL) {
    const r = flood(dist, sx, sy, tol);
    const frac = r.count / total;
    if (frac > R_MAX) break;
    best = {mask: r.mask, frac, tol, how: 'grown'};
    if (frac >= R_MIN) break;
  }
  if (!best || best.frac < R_MIN) {          // flat or ambiguous: use a disc
    const mask = new Uint8Array(total), r = 0.13 * Math.max(g.w, g.h);
    let count = 0;
    for (let y = 0; y < g.h; y++) for (let x = 0; x < g.w; x++)
      if ((x-sx)*(x-sx) + (y-sy)*(y-sy) < r*r) { mask[y*g.w+x] = 1; count++; }
    best = {mask, frac: count/total, tol: null, how: 'disc'};
  }
  best.sx = sx; best.sy = sy;
  return best;
}

function update() {
  if (!D) return;
  const g = D.grid;
  region = growRegion();
  const m = region.mask;

  shifts = [[0,0]]; let magSum = 0;
  D.flows.forEach(fl => {
    let sx = 0, sy = 0, n = 0;
    for (let y = 0; y < g.h; y++) for (let x = 0; x < g.w; x++)
      if (m[y*g.w+x]) { sx += fl[y][x][0]; sy += fl[y][x][1]; n++; }
    sx /= n; sy /= n; magSum += Math.hypot(sx, sy);
    shifts.push([sx, sy]);
  });
  const mx = shifts.reduce((a,s)=>a+s[0],0)/shifts.length;
  const my = shifts.reduce((a,s)=>a+s[1],0)/shifts.length;
  shifts = shifts.map(s => [s[0]-mx, s[1]-my]);

  const T = D.tensor; let a = 0, d = 0, b = 0, n = 0;
  for (let y = 0; y < g.h; y++) for (let x = 0; x < g.w; x++)
    if (m[y*g.w+x]) { a += T.xx[y][x]; d += T.yy[y][x]; b += T.xy[y][x]; n++; }
  a /= n; d /= n; b /= n;
  const tr = a + d, lam = tr/2 - Math.sqrt(Math.max(0, tr*tr/4 - (a*d - b*b)));
  const lock = Math.max(0, Math.min(1, Math.log10(1+lam)/4));
  $('#lock').style.width = (lock*100) + '%';
  $('#lock').style.background = lock > .55 ? 'var(--good)' : lock > .3 ? 'var(--warn)' : 'var(--bad)';
  $('#lockv').textContent = lock.toFixed(2);
  $('#regv').textContent = (region.frac*100).toFixed(1) + '%' +
    (region.how === 'disc' ? ' (no depth edge — fell back to a disc)' : '');
  $('#plx').textContent = (magSum / D.flows.length / D.scale).toFixed(0) + 'px';

  buildRegion();
  padX = Math.ceil(Math.max(...shifts.map(s => Math.abs(s[0]))));
  padY = Math.ceil(Math.max(...shifts.map(s => Math.abs(s[1]))));
  render();
}

function buildRegion() {
  // Mark the anchored region itself, not everything around it: the region is
  // only a few percent of the frame, so tinting the remainder washes out most
  // of the picture you are trying to judge. A bright edge with a faint fill
  // reads clearly without hiding what is underneath.
  const g = D.grid, c = document.createElement('canvas');
  c.width = g.w; c.height = g.h;
  const ctx = c.getContext('2d'), im = ctx.createImageData(g.w, g.h);
  const m = region.mask;
  for (let y = 0; y < g.h; y++) for (let x = 0; x < g.w; x++) {
    const i = y*g.w + x, j = i*4;
    if (!m[i]) { im.data[j+3] = 0; continue; }
    const edge = (x === 0 || !m[i-1]) || (x === g.w-1 || !m[i+1]) ||
                 (y === 0 || !m[i-g.w]) || (y === g.h-1 || !m[i+g.w]);
    im.data[j] = 110; im.data[j+1] = 168; im.data[j+2] = 254;
    im.data[j+3] = edge ? 185 : 45;
  }
  ctx.putImageData(im, 0, 0); regionCanvas = c;
}
function buildDepth() {
  const g = D.grid, c = document.createElement('canvas');
  c.width = g.w; c.height = g.h;
  const ctx = c.getContext('2d'), im = ctx.createImageData(g.w, g.h);
  const last = D.flows[D.flows.length-1];
  let mx = 1e-6;
  for (let y=0;y<g.h;y++) for (let x=0;x<g.w;x++)
    mx = Math.max(mx, Math.hypot(last[y][x][0], last[y][x][1]));
  for (let y=0;y<g.h;y++) for (let x=0;x<g.w;x++) {
    const t = Math.min(1, Math.hypot(last[y][x][0], last[y][x][1])/mx), i = (y*g.w+x)*4;
    im.data[i] = 255*Math.min(1, t*1.8);
    im.data[i+1] = 255*Math.min(1, Math.max(0, t*1.6-.3));
    im.data[i+2] = 255*Math.max(0, 1-t*1.8);
    im.data[i+3] = 150;
  }
  ctx.putImageData(im, 0, 0); depthCanvas = c;
}

// --- rendering -------------------------------------------------------------
function order() {
  let s = imgs.map((_, i) => i);
  if ($('#rev').checked) s = s.reverse();
  if ($('#pp').checked && s.length > 2) s = s.concat(s.slice(1,-1).reverse());
  return s;
}
function loop(t) {
  if (mode === 'wiggle' && D && imgs.length) {
    const seq = order(), ms = 1000 / (+$('#fps').value);
    if (t - t0 > ms) { t0 = t; step = (step+1) % seq.length; cur = seq[step]; render(); }
  }
  requestAnimationFrame(loop);
}
// The wiggle canvas shows the frame minus the alignment margins, so point
// fractions (in FRAME coordinates) shift by that margin to land on what is
// actually being measured.
function toCanvas(fx, fy) {
  return [fx*D.w - padX - shifts[0][0], fy*D.h - padY - shifts[0][1]];
}
function toFrame(cx, cy) {
  return [(cx + padX + shifts[0][0])/D.w, (cy + padY + shifts[0][1])/D.h];
}
function fit(cv, w, h) { if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; } }

function render() {
  if (!S) return;
  const cv = $('#cv'), ctx = cv.getContext('2d');
  if (mode === 'crop') {
    if (!scanImg || !scanImg.complete) return;
    const w = scanImg.naturalWidth, h = scanImg.naturalHeight;
    fit(cv, w, h); ctx.clearRect(0,0,w,h); ctx.drawImage(scanImg, 0, 0);
    const y0 = S.band[0]*S.scale, y1 = S.band[1]*S.scale;
    ctx.fillStyle = 'rgba(0,0,0,.55)';
    ctx.fillRect(0, 0, w, y0); ctx.fillRect(0, y1, w, h - y1);
    ctx.strokeStyle = '#5fd08a'; ctx.lineWidth = 2;
    [y0, y1].forEach(y => { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); });
    ctx.strokeStyle = '#e06c6c';
    S.cuts.forEach(c => { const x = c*S.scale;
      ctx.beginPath(); ctx.moveTo(x,y0); ctx.lineTo(x,y1); ctx.stroke(); });
    return;
  }
  if (!D || !imgs.length) return;
  const w = Math.max(1, Math.round(D.w - 2*padX)), h = Math.max(1, Math.round(D.h - 2*padY));
  fit(cv, w, h);
  const ox = -padX - shifts[cur][0], oy = -padY - shifts[cur][1];
  ctx.clearRect(0,0,w,h);
  ctx.drawImage(imgs[cur], ox, oy);
  if (showDepth && depthCanvas) ctx.drawImage(depthCanvas, ox, oy, D.w, D.h);
  // The region is drawn in the reference frame's position so it stays put while
  // the frames cycle underneath it -- it marks what is being held still.
  const [rx, ry] = [-padX - shifts[0][0], -padY - shifts[0][1]];
  if (showRegion && regionCanvas) ctx.drawImage(regionCanvas, rx, ry, D.w, D.h);
  const [px, py] = toCanvas(point.x, point.y);
  ctx.strokeStyle = '#6ea8fe'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(px, py, 9, 0, 6.284); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(px-15,py); ctx.lineTo(px-4,py); ctx.moveTo(px+4,py); ctx.lineTo(px+15,py);
  ctx.moveTo(px,py-15); ctx.lineTo(px,py-4); ctx.moveTo(px,py+4); ctx.lineTo(px,py+15);
  ctx.stroke();
  [...document.querySelectorAll('#strip .f')].forEach((e,k) =>
    e.className = 'f' + (k === cur ? ' on' : ''));
}

// --- pointer ---------------------------------------------------------------
function rel(e) {
  const cv = $('#cv'), r = cv.getBoundingClientRect();
  return [(e.clientX-r.left)/r.width*cv.width, (e.clientY-r.top)/r.height*cv.height];
}
$('#cv').addEventListener('pointerdown', e => {
  const [cx, cy] = rel(e), T = 10;
  if (mode === 'crop') {
    const cands = [['top', Math.abs(cy - S.band[0]*S.scale)],
                   ['bot', Math.abs(cy - S.band[1]*S.scale)]];
    S.cuts.forEach((c,i) => cands.push([i, Math.abs(cx - c*S.scale)]));
    const best = cands.reduce((a,b) => b[1] < a[1] ? b : a);
    if (best[1] <= T) drag = {crop: best[0]};
  } else if (D) {
    const [u, v] = toFrame(cx, cy);          // click anywhere to place the point
    point = {x: Math.min(1, Math.max(0, u)), y: Math.min(1, Math.max(0, v))};
    drag = {mode: 'point'};
    update();
  }
  if (drag) $('#cv').setPointerCapture(e.pointerId);
});
$('#cv').addEventListener('pointermove', e => {
  if (!drag) return;
  const [cx, cy] = rel(e);
  if (mode === 'crop') {
    if (drag.crop === 'top') S.band[0] = Math.max(0, Math.min(S.band[1]-20, Math.round(cy/S.scale)));
    else if (drag.crop === 'bot') S.band[1] = Math.min(S.scanH, Math.max(S.band[0]+20, Math.round(cy/S.scale)));
    else S.cuts[drag.crop] = Math.max(0, Math.min(S.scanW, Math.round(cx/S.scale)));
    syncCropPanel(); render(); return;
  }
  const [u, v] = toFrame(cx, cy);
  point = {x: Math.min(1, Math.max(0, u)), y: Math.min(1, Math.max(0, v))};
  update();
});
addEventListener('pointerup', () => { if (drag && drag.crop !== undefined) S.cuts.sort((a,b)=>a-b); drag = null; });

$('#dep').onclick = () => { showDepth = !showDepth; markToggle($('#dep'), showDepth); render(); };
function markToggle(el, on) {
  el.style.borderColor = on ? 'var(--accent)' : 'var(--line)';
  el.style.color = on ? 'var(--accent)' : 'var(--fg)';
}
$('#reg').onclick = () => { showRegion = !showRegion; markToggle($('#reg'), showRegion); render(); };
markToggle($('#reg'), showRegion);
addEventListener('keydown', e => {          // quick peek without leaving the mouse
  if (e.key === 'm' || e.key === 'M') { showRegion = !showRegion; markToggle($('#reg'), showRegion); render(); }
});
['fps','mh'].forEach(k => $('#'+k).oninput = () => $('#'+k+'v').textContent = $('#'+k).value);

$('#export').onclick = async () => {
  const b = $('#export'); b.disabled = true; msg('rendering at full resolution…');
  $('#result').textContent = ''; $('#result').className = '';
  try {
    const r = await api('/api/export', {method:'POST', body: JSON.stringify({
      point: [point.x, point.y], fps: +$('#fps').value,
      pingpong: $('#pp').checked, reverse: $('#rev').checked,
      maxHeight: +$('#mh').value, repair: $('#rp').checked})});
    showResult(r);
    msg('wrote ' + r.out);
  } catch (e) { msg(e.message); } finally { b.disabled = false; }
};

// Report what the exporter actually achieved. The preview averages flow while
// export runs ECC, so a frame can fail to lock even when the preview looked
// fine -- without this the page quietly implies a result it did not deliver.
function showResult(r) {
  (r.ccs || []).forEach((cc, k) => {
    const el = $('#cc' + k); if (!el) return;
    el.textContent = 'cc ' + cc.toFixed(2);
    el.className = 'cc ' + (cc >= 0.8 ? 'good' : 'bad');
  });
  const el = $('#result');
  const px = r.size ? r.size.join('×') : '';
  if (r.weak && r.weak.length) {
    el.className = 'bad';
    el.innerHTML = `<b>Frame ${r.weak.join(', ')} did not lock</b> (cc below 0.80)` +
      (r.repaired ? ', so its shift was inferred from the frames that did.'
                  : '. That frame will jump — try clicking somewhere with more ' +
                    'detail, or tick “repair weak frames”.');
  } else {
    el.className = '';
    el.textContent = `All ${r.ccs.length} frames locked. ${px} at ${r.fps.toFixed(2)}fps.`;
  }
}

requestAnimationFrame(loop);
boot().catch(e => msg(e.message));
</script></body></html>
"""


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
