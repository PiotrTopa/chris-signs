#!/usr/bin/env python3
"""Cut JPEG frames of a ride (MJPEG + index, as cache_dets.py) to files: cut_frames.py SLUG OUTDIR START:STOP:STEP
Prints the file paths (one per line). Ride frames are CC BY 3.0 test material, never committed."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_dets import Frames  # noqa: E402

slug, out, spec = sys.argv[1:4]
fr = Frames(slug)
st, sp, stp = (spec.split(":") + ["", ""])[:3]
os.makedirs(out, exist_ok=True)
for k in range(int(st or 0), int(sp) if sp else len(fr), int(stp or 1)):
    p = f"{out}/{k:06d}.jpg"
    if not os.path.exists(p):
        open(p, "wb").write(fr.jpeg(k))
    print(p)
