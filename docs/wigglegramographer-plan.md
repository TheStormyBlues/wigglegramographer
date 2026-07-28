# Wigglegramographer — Project Plan

## Concept
An app that turns a scan of a 3D film-camera negative (e.g. Nimslo 3D) into a
"wigglegram" — a short looping animation that cycles through the camera's four
slightly offset frames to create a 3D depth illusion.

Pipeline at a glance: ingest scan → crop into individual frames → align frames
→ export animated GIF (and optionally MP4 / WebP).

## What the input looks like
Based on a representative sample scan:

- A full "sprocket scan" of 35mm colour negative film (Kodak).
- Sprocket perforations run along the top and bottom edges.
- Edge markings / frame numbers are printed on the film rebate
  (e.g. "KODAK 100-8", 9, 9A, 10, 10A).
- Four image frames (portrait orientation) sit in the central band, separated by
  thin dark vertical gaps (the inter-frame rebate).
- Clear parallax between frames — the subject shifts horizontally relative to the
  background, which is exactly what produces a good wiggle.

Implications for the build:

- Cropping must first isolate the central image band (excluding sprockets and
  edge text), then split that band into the four frames at the dark gaps.
- The dark gaps are high-contrast against frame content, so automatic detection
  should be reliable — but a manual override is worth keeping for messy scans.
- Scans may be slightly skewed; an optional deskew step will help.

## Recommended platform approach
Build in two stages rather than committing to a platform up front:

1. **Python command-line script first.** Fastest path to a working wiggle; lets us
   nail cropping and alignment (the hard parts) without any UI overhead.
2. **Then wrap in a lightweight local web UI** (Gradio or Streamlit). Drag-and-drop
   in the browser, full OpenCV power, nothing to host. This app benefits from
   interactive controls (adjusting the crop, picking the alignment anchor), so a UI
   is worth it — but only once the core pipeline works.

A fully hosted, shareable web app (for other Nimslo shooters) is a good long-term
goal, but save it until the tool reliably makes good wiggles.

Core stack: Python, OpenCV (cropping + alignment), NumPy, imageio / Pillow
(GIF / MP4 / WebP export).

## Technical pipeline
1. **Ingest** — load the scan; optional deskew.
2. **Isolate the image band** — use a row-wise brightness/variance profile to find
   the vertical bounds of the frame band, excluding the sprocket rows and edge-text
   rows.
3. **Crop into four frames** — within the band, use a column-wise brightness/gradient
   profile; the three dark inter-frame gaps show up as valleys, so split at them into
   four frames. Allow manual adjustment.
4. **Align (registration)** — the key quality step. Choose a subject anchor
   (default: central region, or user-selected). Estimate translation (optionally a
   small rotation) between frames using feature matching (ORB/SIFT) or ECC / phase
   correlation on the anchor region, so the subject stays fixed and everything else
   wiggles around it. Note: parallax means only the chosen depth plane aligns — that
   is intended and is what creates the effect.
5. **Sequence & export** — order the frames, play ping-pong (1→2→3→4→3→2→1) so the
   loop is smooth, tune the frame rate, and export a GIF (plus optional MP4 / WebP).

## Roadmap
- **Phase 0 — Recon (done):** gather representative scans and understand the frame
  layout. A sample sprocket scan is in hand.
- **Phase 1 — MVP script (done):** load → crop four frames → ping-pong GIF, no
  alignment yet. Goal: a working end-to-end pipeline you can actually see.
- **Phase 2 — Alignment + robust cropping (done):** anchor-based registration;
  automatic band and gap detection. This was the big quality jump.
- **Phase 3 — UX & polish (in progress):** interactive crop/anchor picking is built,
  along with controls for frame rate, order, and loop style. Still open: automatic
  subject detection, more robust band detection, a local web UI, and MP4 / WebP
  export.

## Open decisions
- **Frame order:** the on-film left-to-right order may need reversing to get the
  wiggle direction right — make it a toggle.
- **Manual vs. automatic:** how much hands-on control over crop and anchor.
- **Sprocket aesthetic:** whether to preserve the film-border look anywhere, or
  always crop to clean frames.
- **Output defaults:** dimensions, frame rate, loop style.

## What we learned building it
- **Profile statistic matters more than threshold.** Both detection stages originally
  averaged a row or column and compared against an adaptive threshold. Averaging is
  the wrong summary: it cannot distinguish "dark all the way across" from "dark in
  places". Switching the column profile to a 90th percentile fixed frames that were
  being cut through the middle. The row profile still has the same weakness, and is
  the main source of over-cropping on dark scenes.
- **Regularity is a free self-check.** The frames sit on a fixed mechanical pitch, so
  their widths must come out near-equal. Comparing them catches a bad split without
  any reference data — correct splits measure under 1%, broken ones several percent.
- **Failures must be loud.** ECC returns a correlation score that the first cut of
  the code discarded, so a frame that failed to register was written out unaligned
  and the run still reported success. Surfacing that score turned the most confusing
  failure mode into a one-line diagnosis.
- **Alignment is more fragile than it looks.** A five-pixel change in where the cuts
  land was enough to move a frame's estimated shift by 40px. Trust the correlation
  score over the shift values when judging whether a frame really locked.

## Next step
Make band detection as robust as gap detection. Row standard deviation is the
candidate: every row inside the band crosses three black inter-frame gaps, so it has
high horizontal variance regardless of how dark the scene is. Early testing recovers
most of the height lost on dark scans, but it perturbs the scans that already work,
so it needs validating across the whole roll before adoption.
