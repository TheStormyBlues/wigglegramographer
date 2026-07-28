# Wigglegramographer

Turn a scan of a Nimslo 3D film negative into a looping wigglegram GIF — auto-crops
the four frames and aligns them so your subject stays locked while the background
wiggles.

Wigglegramographer takes a flatbed "sprocket scan" of a 3D film-camera negative
and turns it into a wigglegram: a short looping animation that cycles through the
camera's four slightly-offset frames to create a 3D depth illusion. It isolates the
image band, splits it into the individual frames at the inter-frame gaps, and uses
anchor-based registration to lock your subject in place so everything else
parallaxes around it. Built with Python and OpenCV.

## Features

- Auto-isolates the image band (skips sprockets and edge text) from a full scan.
- Splits the band into frames at the dark inter-frame gaps, with an even-split fallback.
- Anchor-based masked-ECC alignment so the subject stays fixed and the background wiggles.
- Interactive picking: drag the crop and the subject box, then replay the choice as flags.
- Reports a per-frame lock quality score, so a frame that fails to register says so.
- Seamless ping-pong looping, adjustable frame rate, and reversible wiggle direction.
- Debug mode that dumps the detection overlay and a per-frame contact sheet.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Basic: aligned wiggle from a scan
python wigglegram.py scan.jpg

# See how it detected the band, gaps, and anchor
python wigglegram.py scan.jpg --debug

# Raw, unaligned wiggle (before/after comparison)
python wigglegram.py scan.jpg --align none

# Interactive: fix the crop, then drag a box over your subject
python wigglegram.py scan.jpg --pick-anchor

# Tune the subject box by hand (fractions: centre-x, centre-y, width, height)
python wigglegram.py scan.jpg --anchor 0.5,0.7,0.3,0.4

# Speed and direction
python wigglegram.py scan.jpg --fps 10 --reverse
```

### Flags

| Flag | What it does |
| --- | --- |
| `--align translation\|euclidean\|none` | Alignment mode. `translation` is the default; `euclidean` adds rotation and is experimental. |
| `--anchor cx,cy,w,h` | Subject box as fractions of the frame. |
| `--pick-anchor` | Interactive: adjust the crop, then drag the subject box. |
| `--band y0,y1` | Override the detected image band, in pixels. |
| `--cuts x0,...,xN` | Override the frame cut positions, in pixels. |
| `--repair-weak` | Infer shifts for frames that fail to lock from the ones that did. |
| `--frames N` | Number of frames on the scan (default 4). |
| `--fps`, `--reverse`, `--no-pingpong` | Playback speed, direction, and loop style. |
| `--max-height`, `--inset` | Output height cap and per-frame edge trim. |
| `--debug` | Write the detection overlay and contact sheet alongside the GIF. |

## Interactive picking

`--pick-anchor` runs two steps:

1. **Crop** — the whole scan, with the detected band edges in green and the frame
   cuts in red. Drag any line to correct it. A live width-spread readout tells you
   when the cuts are evenly spaced, which is what a correct split looks like.
2. **Anchor** — the first cropped frame. Drag a box over your subject; that region,
   and only that region, drives the alignment.

Both steps print their result back as flags:

```
crop      : --band 805,3140 --cuts 238,2182,4154,6120,8061
anchor    : --anchor 0.489,0.556,0.420,0.444
```

Paste those into a later run to reproduce the same result without the windows —
handy for batch processing a roll once you've dialled one frame in.

## How it works

1. **Isolate the band** — a row-brightness profile finds the vertical bounds of the
   frame band, excluding the dark sprocket and edge-text rows.
2. **Split into frames** — a column profile locates the dark inter-frame gaps and
   cuts the band into individual frames. Each column is scored by its 90th
   percentile rather than its mean: a real gap is dark for the *full* height of the
   band, whereas dark image content (a doorway, a shadow) has bright pixels above or
   below it in the same column. A mean cannot tell those apart, and the cuts end up
   slicing through the middle of the pictures.
3. **Align** — a masked ECC registration estimates the shift that keeps the subject
   inside the anchor box fixed; only that region drives the estimate, so the busy
   background is ignored. Parallax means only the subject's depth plane aligns —
   that's what creates the effect.
4. **Sequence & export** — frames play ping-pong (1→2→3→4→3→2) for a seamless loop,
   exported as a GIF.

## When it goes wrong

The tool checks its own work and tells you which stage failed.

**"uneven frame widths"** — the four frames sit on a fixed mechanical pitch, so
their widths should be near-identical. A spread above 2% means the cuts missed the
gaps and your frames straddle two pictures. Run with `--debug` and look at the red
cut lines in `*_debug_detect.png`, then fix them with `--pick-anchor` or `--cuts`.

**"frame N did not lock onto the anchor"** — each frame prints a `cc` score. Above
0.85 is a solid lock; below 0.80 that frame will visibly jump. Usually the anchor
box is sitting on something featureless — blank wall, floor, or a dark silhouette
with no internal detail. Re-run with `--pick-anchor` and choose a smaller,
high-contrast detail such as a face or an eye. If you'd rather not re-pick,
`--repair-weak` will infer the missing shift from the frames that did lock.

**Output looks over-cropped** — band detection clips the top and bottom of dark
scenes, because a dark row drags the row average below the detection threshold.
Use `--pick-anchor` and drag the green band edges outward, or pass `--band y0,y1`
directly.

**The GIF won't animate on a phone** — GIF keeps its frame delay in a Graphic
Control Extension, and a file without one plays at whatever speed the decoder
guesses, or not at all. Browsers are forgiving; Android is not. Every GIF written
here carries an explicit delay, a do-not-dispose disposal method, and an infinite
loop block. Note that the format stores delay in hundredths of a second, so the
frame rate is rounded to that grid — `--fps 8` really plays at 8.33, and anything
above 50 is capped. The run prints the effective rate.

**Colours shimmer between frames** — shouldn't happen here, but worth knowing why.
A GIF frame can carry its own 256-colour table, and if each frame is quantised on
its own those tables drift, so flat areas crawl even where the picture is identical.
Wigglegramographer derives one palette from every frame and applies it to all of
them, which holds the colours still for a couple of KB.

## Roadmap

- **Phase 1 — MVP:** crop four frames → ping-pong GIF. ✅
- **Phase 2 — Alignment:** anchor-based registration + robust cropping. ✅
- **Phase 3 — UX & polish:** interactive crop and anchor picking ✅; still to come:
  automatic subject detection, more robust band detection, a local web UI, and
  MP4 / WebP export.

See `docs/wigglegramographer-plan.md` for the full plan.
