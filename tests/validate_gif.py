#!/usr/bin/env python3
"""Check exported GIFs against the GIF89a spec and common decoder expectations.

This exists because a GIF can be badly broken while looking completely normal:
the original export wrote no Graphic Control Extension at all, so the files had
no frame delay and no disposal method. They still looped in a browser, and
nothing short of parsing the bytes revealed it.

Walks the block structure properly (data sub-blocks are skipped, not scanned,
so byte patterns inside compressed image data cannot be mistaken for headers).

    python tests/validate_gif.py                 # checks outputs/*.gif
    python tests/validate_gif.py a.gif b.gif     # or specific files
"""
import glob
import os
import sys

import numpy as np


def parse(b):
    """Walk the GIF block structure into a structured description."""
    d = {"sig": b[0:6], "w": b[6] | (b[7] << 8), "h": b[8] | (b[9] << 8),
         "gct": bool(b[10] & 0x80), "gct_size": 2 ** ((b[10] & 0x07) + 1),
         "bg": b[11], "frames": [], "loop": None, "loop_before_image": None,
         "trailer": False, "trailing_bytes": 0, "palettes": []}
    i = 13
    if d["gct"]:
        n = d["gct_size"]
        d["global_palette"] = np.frombuffer(b[13:13 + 3 * n], np.uint8) \
                                .reshape(-1, 3).astype(np.int16)
        i += 3 * n
    else:
        d["gct_size"], d["global_palette"] = 0, None
    pending, seen_image = None, False

    def skip(j):
        while b[j] != 0:
            j += 1 + b[j]
        return j + 1

    while i < len(b):
        if b[i] == 0x3B:
            d["trailer"] = True
            d["trailing_bytes"] = len(b) - i - 1
            break
        if b[i] == 0x21:
            label = b[i + 1]
            if label == 0xF9:
                p = b[i + 3]
                pending = {"delay": b[i + 4] | (b[i + 5] << 8),
                           "disposal": (p >> 2) & 0x07,
                           "transparent": bool(p & 0x01),
                           "user_input": bool(p & 0x02)}
            elif label == 0xFF and b[i + 3:i + 14] == b"NETSCAPE2.0":
                d["loop"] = b[i + 17] | (b[i + 18] << 8)
                d["loop_before_image"] = not seen_image
            i = skip(i + 2)
        elif b[i] == 0x2C:
            seen_image = True
            p = b[i + 9]
            f = {"left": b[i + 1] | (b[i + 2] << 8), "top": b[i + 3] | (b[i + 4] << 8),
                 "w": b[i + 5] | (b[i + 6] << 8), "h": b[i + 7] | (b[i + 8] << 8),
                 "lct": bool(p & 0x80), "interlace": bool(p & 0x40), "gce": pending}
            d["frames"].append(f)
            pending = None
            if f["lct"]:
                n = 2 ** ((p & 0x07) + 1)
                d["palettes"].append(np.frombuffer(b[i + 10:i + 10 + 3 * n], np.uint8)
                                       .reshape(-1, 3).astype(np.int16))
                i += 10 + 3 * n
            else:
                d["palettes"].append(d["global_palette"])
                i += 10
            i = skip(i + 1)
        else:
            d["error"] = f"unexpected block 0x{b[i]:02X} at offset {i}"
            break
    return d


def check(path):
    d = parse(open(path, "rb").read())
    res = []

    def ok(cond, label, detail=""):
        res.append((bool(cond), label, detail))

    ok(d["sig"] == b"GIF89a", "header is GIF89a", d["sig"].decode("latin-1"))
    ok("error" not in d, "block structure parses", d.get("error", ""))
    ok(d["trailer"], "trailer 0x3B present")
    ok(d["trailing_bytes"] == 0, "no bytes after trailer", f"{d['trailing_bytes']} extra")
    ok(d["gct"], "global colour table present", f"{d['gct_size']} entries")
    ok(d["bg"] < max(1, d["gct_size"]), "background index within table",
       f"bg={d['bg']} of {d['gct_size']}")
    ok(d["loop"] is not None, "NETSCAPE loop block present")
    ok(d["loop"] == 0, "loop count is infinite", f"loop={d['loop']}")
    ok(d["loop_before_image"] is True, "loop block precedes the first image")
    ok(len(d["frames"]) > 1, "more than one frame", f"{len(d['frames'])} frames")

    gces = [f["gce"] for f in d["frames"]]
    ok(all(g is not None for g in gces), "every frame has a GCE",
       f"{sum(g is not None for g in gces)}/{len(gces)}")
    if all(g is not None for g in gces):
        delays = sorted({g["delay"] for g in gces})
        disps = sorted({g["disposal"] for g in gces})
        ok(all(g["delay"] >= 2 for g in gces),
           "delay >= 2cs (decoders clamp below this)", f"delays={delays}")
        ok(len(delays) == 1, "delay consistent across frames", str(delays))
        ok(set(disps) <= {1, 2}, "disposal is a defined replace mode", str(disps))
        ok(len(disps) == 1, "disposal consistent across frames", str(disps))
        ok(not any(g["transparent"] for g in gces),
           "no transparency flag on opaque frames")
        ok(not any(g["user_input"] for g in gces), "user-input flag clear")

    ok(all(f["left"] == 0 and f["top"] == 0 for f in d["frames"]),
       "all frames at offset 0,0")
    ok(all(f["w"] == d["w"] and f["h"] == d["h"] for f in d["frames"]),
       "all frames fill the logical screen", f"screen {d['w']}x{d['h']}")
    ok(not any(f["interlace"] for f in d["frames"]), "no interlacing")

    # Per-frame palettes that drift make flat areas crawl. Frame count, local
    # table count and file size all look identical either way, so compare the
    # colour tables themselves.
    drift = 0
    for a, b_ in zip(d["palettes"], d["palettes"][1:]):
        if a is not None and b_ is not None:
            n = min(len(a), len(b_))
            drift = max(drift, int(np.abs(a[:n] - b_[:n]).max()))
    ok(drift == 0, "palette is stable across frames", f"max channel drift {drift}")
    return d, res


def main():
    paths = sys.argv[1:] or sorted(glob.glob(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "outputs", "*.gif")))
    if not paths:
        print("no GIFs found; run wigglegram.py first")
        return 1
    failed = 0
    for p in paths:
        d, res = check(p)
        bad = [r for r in res if not r[0]]
        failed += bool(bad)
        print(f"{os.path.basename(p):<30} {d['w']}x{d['h']} {len(d['frames'])}f  "
              f"{'PASS' if not bad else f'{len(bad)} FAIL'}")
        for good, label, detail in bad:
            print(f"      FAIL  {label}   {detail}")
    print(f"\n{len(paths)} file(s) checked, {failed} with failures")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
