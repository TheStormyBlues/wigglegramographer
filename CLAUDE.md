# Wigglegramographer

Converts a flatbed "sprocket scan" of a 3D film-camera negative (Nimslo 3D, four
frames) into a looping wigglegram GIF. Python + OpenCV, single-file CLI.

## Pipeline
ingest scan → isolate central image band → split into 4 frames at the dark
inter-frame gaps → align frames onto a subject anchor → ping-pong sequence → GIF.

## Layout
- `wigglegram.py` — the whole tool (detection, cropping, alignment, export, CLI).
- `samples/` — test scans (`_DSC5457`–`_DSC5472.jpg`, colour negative, person
  holding a black cat). `_DSC5466.jpg` is the cleanest reference asset.
  Note `samples/` and generated output are gitignored, so these are local-only.
- `docs/wigglegramographer-plan.md` — full project plan and roadmap.

## Run it
    pip install -r requirements.txt
    python wigglegram.py samples/_DSC5466.jpg --debug

Key flags: `--align translation|euclidean|none`, `--anchor cx,cy,w,h` (fractions),
`--pick-anchor`, `--band y0,y1`, `--cuts x0,...,xN`, `--repair-weak`, `--fps`,
`--reverse`, `--no-pingpong`, `--frames`, `--debug`.

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
- `--repair-weak` (opt-in) replaces weak frames' shifts with a linear fit through
  the frames that did lock. The four lenses sit on a fixed pitch, so shift is
  linear in frame index for any one depth plane. Needs >= 2 good frames; with
  fewer it declines and keeps the measured values.

## Known rough edges
- `detect_band` clips the top and bottom of dark scenes, because a dark row drops the
  row mean below threshold. This, not the alignment crop, is what makes output look
  over-cropped: the alignment intersection costs only 2-10% of area, while band
  clipping costs `_DSC5472` 50% of its height, `_DSC5468` 35% and `_DSC5465` 30%
  (against a ~2350px norm). Fix it per-scan with `--pick-anchor` or `--band`.
  Row standard deviation is a promising automatic fix — every band row crosses three
  black gaps, so it has high horizontal variance whatever the scene — and in testing
  it took `_DSC5465` from 1650 to 2352px, `_DSC5468` to 1898 and `_DSC5472` to 1395.
  Not yet adopted, since it also nudges the good scans and alignment is edge-sensitive.
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

## Next up (Phase 3)
- Fix `_DSC5472`: band detection, not the cuts. Candidate is a grid fit — autocorrelate
  the column profile for the frame pitch, then solve for the phase that puts the three
  gaps in the darkest columns, making frame widths even by construction.
- Adopt row-std band detection so the crop step is rarely needed (see rough edges).
- Auto subject detection, so the anchor does not have to be picked by hand.
- Optional local web UI (Gradio or Streamlit) for drag-and-drop + interactive anchor.
- MP4 / WebP export alongside GIF.

## Conventions
- Keep it a single-file CLI for now; don't add a framework prematurely.
- Detection is profile-based (row/column brightness) — prefer adaptive thresholds
  over hardcoded pixel coordinates so it generalizes across scans.
- Always preserve the `--align none` path for debugging raw output.
