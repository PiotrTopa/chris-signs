#!/usr/bin/env python3
"""Run one sign variant on cached detections: schedule -> signs_track_replay -> OUT.jsonl (lane L24-S).

  variant.py SLUG OUT.jsonl --hz 5 [--phase 0] [--view2 B1 --every2 2] [--extra LW,LZ] [-- tracker args]

--hz: processed frames per second of the 20 fps ride (frame n = floor(n * 20 / hz) + phase, as signd taking the
newest frame every 1/hz s). --view2 NAME: every --every2-th processed frame uses that window instead of the whole
frame (signd --roi2; frames whose window detections are not cached fall back to the whole frame). --extra: on every
processed frame whose detections exist in those sources, they are merged (alternating between the listed sources),
i.e. the light detector's `traffic sign` boxes handed to the sign tracker.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

H = os.path.expanduser("~")
RECT = {"B1": "380,192,1020,576", "B2": "213,104,1067,616", "LW": "0,0,1280,447", "LZ": "160,90,1120,410"}
TOOL = f"{H}/datasets/l24s/build/signs_track_replay"


def main():
    argv = sys.argv[1:]
    extra_args = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra_args = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("out")
    ap.add_argument("--hz", type=float, default=5)
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--view2")
    ap.add_argument("--every2", type=int, default=2)
    ap.add_argument("--extra", default="")
    ap.add_argument("--lights", default="", help="LW,LZ: the light thread's sign proposals as tracker updates of their "
                    "own (as signd --lights-signs) on every cached frame of those sources, alternating")
    ap.add_argument("--no-det", action="store_true", help="no sign-detector frames at all: the light thread's "
                    "sign proposals (--lights) are the only tracker updates (signd --det none, lane L48)")
    ap.add_argument("--cache", default=f"{H}/datasets/l24s/cache")
    a = ap.parse_args(argv)
    base = f"{a.cache}/{a.slug}"
    nfr = int(np.load(f"{base}.A.npz")["frames"].max()) + 1
    have = {}
    lights = [e for e in a.lights.split(",") if e]
    for nm in ([a.view2] if a.view2 else []) + [e for e in a.extra.split(",") if e] + lights:
        have[nm] = set(int(f) for f in np.load(f"{base}.{nm}.npz")["frames"])
    lines, n, ne = [], 0, 0
    extras = [e for e in a.extra.split(",") if e]
    while not a.no_det:
        f = int(np.floor(n * 20.0 / a.hz + 1e-9)) + a.phase
        if f >= nfr:
            break
        src = "A"
        if a.view2 and n % a.every2 == 0 and f in have[a.view2]:
            src = a.view2
        if extras:
            e = extras[ne % len(extras)]
            if f in have[e]:
                src += "+" + e
                ne += 1
        lines.append(f"{f} {src}")
        n += 1
    if lights:  # light-thread updates: 5 Hz alternating views on the frames where they are cached
        fr_l = sorted(set().union(*[have[l] for l in lights]))
        if a.no_det and a.phase:  # the other phase of the two views (which view sees which frame)
            fr_l = fr_l[a.phase:]
        sl = [(f, 1, f"{f} {lights[i % len(lights)]}") for i, f in enumerate(fr_l) if f in have[lights[i % len(lights)]]]
        sg = [(int(x.split()[0]), 0, x) for x in lines]
        lines = [x for _, _, x in sorted(sg + sl)]
    sched = a.out + ".sched"
    open(sched, "w").write("\n".join(lines) + "\n")
    cmd = [TOOL, "--dets", f"A={base}.A.dets", "--schedule", sched, "--cls-names", f"{H}/l12/cls1/classes.txt",
           "--class-min-score", "regulatory--no-entry=0.71", "--dist-match-new", "2.0"]
    for nm in have:
        cmd += ["--dets", f"{nm}={base}.{nm}.dets"]
        if nm in RECT:
            cmd += ["--rect", f"{nm}={RECT[nm]}"]
    cmd += extra_args
    with open(a.out, "w") as fo:
        subprocess.run(cmd, stdout=fo, check=True)


if __name__ == "__main__":
    main()
