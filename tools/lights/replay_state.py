#!/usr/bin/env python3
"""State agreement on top of replay_score.py: for every reference light-frame of a lit chain that a run covers,
is the run's reported (tracked) state the chain's majority colour?

replay_state.py --ref F:S ... --run NAME=F:S ...   (same arguments as replay_score.py)
Prints per run: covered light-frames, same colour %, other colour %, reported off/unknown %.
"""
import argparse, bisect, collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_score as rs

COL = ("red", "yellow", "green", "red_yellow")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs="+", required=True); ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--fps", type=float, default=20.0); ap.add_argument("--score", type=float, default=0.35)
    ap.add_argument("--min-s", type=float, default=0.8); ap.add_argument("--stale-s", type=float, default=0.5)
    a = ap.parse_args()
    chains, _ = rs.reference(a.ref, a.score, a.min_s, a.fps)
    lit = [c for c in chains if c["lit"]]
    for c in lit:
        cnt = collections.Counter("red" if s == "red_yellow" else s for s in c["states"] if s in COL)
        c["col"] = cnt.most_common(1)[0][0]
    for r in a.run:
        name, _, spec = r.partition("=")
        msgs = rs.load(spec); idx = [fr for fr, _ in msgs]
        n = same = other = dark = 0
        for c in lit:
            for fr, box in c["frames"].items():
                k = bisect.bisect_right(idx, fr) - 1
                if k < 0 or (fr - idx[k]) / a.fps > a.stale_s: continue
                m = [li for li in msgs[k][1].get("lights") or [] if rs.near(box, li["box"])]
                if not m: continue
                st = m[0].get("state")
                st = "red" if st == "red_yellow" else st
                n += 1
                if st == c["col"]: same += 1
                elif st in COL: other += 1
                else: dark += 1
        print("%-10s covered light-frames %5d  same colour %5.1f %%  other colour %4.1f %%  off/unknown %5.1f %%" % (
            name, n, 100 * same / max(1, n), 100 * other / max(1, n), 100 * dark / max(1, n)))

main()
