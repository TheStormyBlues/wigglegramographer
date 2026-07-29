# Wigglegramographer

Converts a flatbed "sprocket scan" of a 3D film-camera negative (Nimslo 3D, four
frames) into a looping wigglegram GIF. Python + OpenCV, single-file CLI.

## Pipeline
ingest scan → isolate central image band → split into 4 frames at the dark
inter-frame gaps → align frames onto a subject anchor → ping-pong sequence → GIF.

## Layout
- `wigglegram.py` — the engine and CLI (detection, cropping, alignment, export).
- `ui.py` — local web UI, a client of the engine. Stdlib `http.server` plus one
  embedded HTML page, no new dependencies. Deliberate break from the single-file
  rule: the engine stays headless and testable, the UI imports it. Two stages
  (Crop / Wiggle) covering the whole workflow: open, crop, anchor, tune, export.
  `python ui.py` with no argument starts empty and takes a file from the page.
- `tests/` — picker state machines and GIF conformance; see Tests below.
- `samples/` — test scans (`_DSC5457`–`_DSC5472.jpg`, colour negative, person
  holding a black cat). `_DSC5466.jpg` is the cleanest reference asset.
  Note `samples/` and generated output are gitignored, so these are local-only.
- `docs/wigglegramographer-plan.md` — full project plan and roadmap.

## Run it
    pip install -r requirements.txt
    python wigglegram.py samples/_DSC5466.jpg --debug

Key flags: `--align translation|euclidean|none`, `--anchor-point cx,cy`,
`--anchor cx,cy,w,h` (fractions), `--auto-anchor`, `--pick-anchor`, `--band y0,y1`,
`--cuts x0,...,xN`, `--repair-weak`, `--fps`, `--reverse`, `--no-pingpong`,
`--frames`, `--debug`.

## Current state
- Phase 1 (crop → ping-pong GIF) and Phase 2 (anchor-based masked-ECC alignment)
  are working. 15 of the 16 sample scans now split correctly (`_DSC5472` is the holdout).
- `detect_frames` profiles each column by its 90th percentile, not its mean. A real
  inter-frame gap is dark for the full band height; dark image content (a doorway) has
  bright pixels above or below it in the same column. Using the mean, dark content
  scored darker than the real gaps and cuts landed mid-frame, so every "frame" straddled
  two images. This fixed `_DSC5460` (7.8% → 0.7% width spread) and `_DSC5461` (3.7% → 0.7%).
- `width_spread()` is a self-check: the frames sit on a fixed mechanical pitch, so their
  widths must be near-equal. Correct splits measure <1%, broken ones several percent.
  Runs above 2% print a warning instead of silently emitting a garbage GIF.
- Default alignment is `translation`; `euclidean` is experimental (adds noisy rotation).
- `--pick-anchor` runs two interactive steps: first `pick_crop` on the whole scan
  (drag the green band edges and the red cut lines, with a live width-spread
  readout), then `pick_anchor` on frame 1 to drag the subject box. Both print
  their result back as flags (`--band`, `--cuts`, `--anchor`) so an interactive
  session can be replayed non-interactively. The crop step runs even under
  `--align none`, since cropping is independent of alignment.
- `align_frames` returns per-frame ECC correlation scores. cc >= 0.85 is a solid
  lock, below 0.80 the frame is reported as a weak lock and will visibly jump.
  This is the signal that used to be discarded, which is why bad frames were silent.
- `export_gif` writes through Pillow, not imageio, so the Graphic Control Extension
  is under our control. The original call passed `duration=1.0/fps` to imageio, whose
  Pillow plugin expects **milliseconds** — 0.125 rounded to 0 and Pillow then emitted
  no GCE at all, so the GIFs carried no frame delay and no disposal method. They
  looped in browsers (the NETSCAPE block was present) but would not play on Android.
  Delay is now quantised to centiseconds, floored at 2cs, with disposal set to
  do-not-dispose. `export_gif` returns the effective fps, since the centisecond grid
  rounds it (8fps becomes 8.33, and anything above 50fps clamps).
- One palette is derived from all frames and applied to each, rather than letting
  every frame quantise independently. Independent quantisation gave each frame its
  own colour table drifting by up to 164/255 per channel between consecutive frames,
  which makes flat areas crawl even where the picture is identical. Judge this by
  comparing the colour tables themselves — frame count, local-table count and file
  size all look unchanged, so they hide the problem.
- ECC is **seeded from the parallax field** by default. It is a local optimiser and
  was starting from the identity — "no shift at all" — while real shifts here run
  past 100px. `flow_field()` computes dense flow once (~0.4s) and `_region_seeds()`
  turns it into a starting guess. Across the roll this took weak frames 12 -> 10
  with no scan worse, and fixed `_DSC5460` outright (min cc 0.475 -> 0.866), which
  had needed a hand-picked anchor since the first batch. 11 of 16 scans are now
  clean with the default anchor. Pass `flow=` to `align_frames` to reuse a field
  already computed rather than paying for it twice.
- `--anchor-point cx,cy` anchors on a point instead of a box. `region_from_point()`
  grows a region outward while parallax stays close to the clicked point's, so it
  follows the subject and stops at its silhouette; a rectangle straddles depth
  planes and asks ECC to satisfy two motions at once. Masked ECC already accepted
  arbitrary masks, so this needed no change to the alignment core — the box was
  only ever one way of filling a mask in.
- **`_REGION_MIN = 0.02` is correct — do not raise it.** Measured *without* seeding,
  point regions looked worse than the box (14 weak vs 12) and the damage tracked
  region size, which pointed at the floor being too low. That reading was wrong: a
  sweep with seeding on gives box=10, floor 0.02=10, floor 0.08=11, floor 0.15=11.
  Small regions were failing because ECC searched from zero with few pixels, not
  because they were small. Seeded, they are fine. Forcing regions *larger* brings
  back the depth-mixing the point anchor exists to avoid — `_DSC5471` goes from 0
  weak frames at a 7% region to 3 weak at 16%. Small and depth-consistent is the
  right shape; it just needs a seeded optimiser.
- Point regions and the box tie at 10 weak frames when the point is dropped blindly
  at the frame centre, which is pessimistic — a deliberate click does better. On
  `_DSC5472` with a point on the subject the region beat the box 1 weak frame to 3.
- General lesson from that pair of experiments: measure a change *after* the other
  fixes it interacts with, not before. Judged against unseeded ECC the region idea
  looked like a regression worth reverting.
- `--auto-anchor` (opt-in) ranks candidate boxes by the smaller eigenvalue of the
  structure tensor — the quantity that governs how well-constrained a translation
  estimate is, so a region of purely horizontal edges scores low despite high
  contrast — then verifies candidates against the real ECC lock scores. The default
  box is one of the candidates, so it can only match or beat it: measured across the
  roll, weak frames go 12 -> 7 with no scan regressing, and 9 of 16 need no search.
- **Known limit of `--auto-anchor`: a better lock is not the same as the right
  subject.** The anchor also selects which depth plane freezes. On `_DSC5463` it
  picks the drinking glasses behind the subject (cc 0.430 -> 0.791) which locks the
  background and makes the *person* wiggle — backwards for a wigglegram. The centre
  prior is not enough when the subject is off-centre. The promising fix is a depth
  prior: parallax magnitude is a direct proxy for distance, so foreground regions
  shift more between lenses. Prefer candidates that lock well *and* show large
  shift, rather than lock quality alone. This is why the flag stays opt-in.
- `--repair-weak` (opt-in) replaces weak frames' shifts with a linear fit through
  the frames that did lock. The four lenses sit on a fixed pitch, so shift is
  linear in frame index for any one depth plane. Needs >= 2 good frames; with
  fewer it declines and keeps the measured values.

## Known rough edges
- `detect_band` now profiles rows by horizontal standard deviation at `rel_thr=0.25`,
  not brightness. Every band row crosses three black gaps so its variance is high
  whatever the scene, whereas a dark row drags the *mean* under the threshold and the
  band gets clipped. That clipping, not the alignment crop, was the real source of
  over-cropping: alignment intersection costs only 2-10% of area, while brightness
  band detection cost `_DSC5465` 30% of its height, `_DSC5468` 35% and `_DSC5472` 50%.
  Adopting variance took `_DSC5465` 1650 -> 2358px, `_DSC5468` 1532 -> 1920 and
  `_DSC5472` 1162 -> 1646, with total weak-lock frames across the roll going 14 -> 12.
  Thresholds of 0.35 and 0.45 were both worse; 0.45 breaks `_DSC5460` and `_DSC5462`.
- `_DSC5460` needs a hand-picked anchor: the default box lands on blank floor, and its
  black cat is a featureless silhouette ECC cannot lock onto (frame 4 cc=0.47).
  `--anchor 0.60,0.40,0.30,0.30` (the white cat) gets every frame to cc >= 0.89.
- Alignment is sensitive to tiny crop changes. `_DSC5464` went from a symmetric ramp
  to a lumpier one when its cuts moved by <=5px during the percentile switch. Treat a
  low cc, not the shift values, as the authority on whether a frame really locked.
- `_DSC5472` is still unfixed: 5.9% width spread and only one frame locks, so
  `--repair-weak` correctly declines. It needs the band sorted out first.
- Reading `--debug` shifts: frame 1 is the ECC reference and translation mode
  mean-centres the set, so frame 1's printed `dx` is `-mean`, not a measurement.
  A healthy scan ramps monotonically from `-X` to `+X` and sums to zero.

## UI notes
- The realtime anchor preview exists because ECC costs **~30s per anchor** at full
  resolution — measured, not estimated. Interactive selection through ECC is not
  possible, so `ui.py` precomputes the dense parallax field once (~0.4s) and every
  anchor's shift becomes a box average over that field. The field is shipped to the
  browser as a coarse grid, so dragging costs no server round-trip.
- The preview is deliberately approximate: box-mean flow tracks ECC to a median of
  ~9px at full resolution on well-locked frames, which is ~2px at display scale.
  Export re-runs real ECC. Where they disagree badly it is usually ECC that is
  wrong — on `_DSC5460` ECC reports dx=-46 where flow says +104, and that frame is
  a known failed lock.
- The depth overlay is flow magnitude, since parallax is a proxy for distance. It
  is the visual answer to the `--auto-anchor` background problem: you can see which
  regions are foreground before choosing one.
- Endpoints: `GET /api/scanview[?force=1]` (crop stage; `force` re-runs detection),
  `POST /api/upload` (raw body + `X-Filename`, avoids multipart parsing),
  `POST /api/open` (server-side path), `POST /api/prepare` (band/cuts -> frames and
  parallax field), `POST /api/export`. Server-side `STATE` holds path/band/cuts, so
  export always uses exactly the crop the page is showing.
- Both stages draw onto a **single** canvas — the frame and its overlay share one
  surface. An earlier version used a second canvas positioned over the first, which
  gave two bugs: it inherited `background:#000` from the shared `canvas` selector
  and hid the preview entirely, then once transparent its `inset:0` box did not
  match the rescaled canvas underneath and left dead space around the picture.
- Point fractions are in *frame* coordinates but the wiggle canvas shows the frame
  minus the alignment margins, so `toCanvas`/`toFrame` convert. Skipping that puts
  the marker a few percent away from what is actually being measured.
- The anchor mask highlights the region itself (bright edge, faint fill) rather than
  dimming everything around it. The region is a few percent of the frame, so tinting
  the remainder washed out most of the picture. Toggle with the button or **M**.
- The browser grows the region on the coarse grid with the same tolerance ladder as
  `region_from_point()`, so preview and export agree on the mask.
- **Preview and export still use different algorithms** — the preview averages flow,
  export runs ECC. Seeding narrowed the gap (export now starts where the preview
  says and refines) but a scan where ECC fails outright can still diverge from what
  the page showed. The page does not yet report post-export `cc`, so it can imply a
  result the exporter did not deliver.

## Tests
    python tests/test_pickers.py     # picker state machines + coordinate maths
    python tests/validate_gif.py     # GIF89a conformance of outputs/*.gif
Both exit non-zero on failure. `validate_gif.py` exists because a GIF can be badly
broken while looking normal — see the export notes above.

## Next up (Phase 3)
- Fix `_DSC5472`: still 5.1% width spread even with the taller band. Candidate is a
  grid fit — autocorrelate the column profile for the frame pitch, then solve for the
  phase that puts the three gaps in the darkest columns, making widths even by
  construction.
- Give `--auto-anchor` a depth prior so it prefers the foreground subject, not just
  the firmest lock (see the `_DSC5463` case above), then consider making it default.
- Optional local web UI (Gradio or Streamlit) for drag-and-drop + interactive anchor.
- MP4 / WebP export alongside GIF.

## Conventions
- Keep it a single-file CLI for now; don't add a framework prematurely.
- Detection is profile-based (row/column brightness) — prefer adaptive thresholds
  over hardcoded pixel coordinates so it generalizes across scans.
- Always preserve the `--align none` path for debugging raw output.
