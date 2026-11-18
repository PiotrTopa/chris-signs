#!/usr/bin/env python3
"""Merge cache_dets.py outputs of consecutive frame ranges: merge_parts.py OUT_BASE PART_BASE... (per view name)
e.g. merge_parts.py cache/krakow-2024.A parts/krakow-2024.p000.A parts/krakow-2024.p001.A ..."""
import struct
import sys

import numpy as np

out, parts = sys.argv[1], sys.argv[2:]
zs = [np.load(p + ".npz") for p in parts]
np.savez_compressed(out + ".npz", **{k: np.concatenate([z[k] for z in zs]) for k in zs[0].files})
nc, nfr, body = None, 0, []
for p in parts:
    b = open(p + ".dets", "rb").read()
    magic, ver, c, n = struct.unpack("<4siii", b[:16])
    assert magic == b"L24D"
    nc = c
    nfr += n
    body.append(b[16:])
with open(out + ".dets", "wb") as f:
    f.write(struct.pack("<4siii", b"L24D", 1, nc, nfr))
    for x in body:
        f.write(x)
