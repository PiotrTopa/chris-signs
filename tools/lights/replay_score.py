#!/usr/bin/env python3
"""How much of the traffic lights that are there does a signd --lights configuration report?

A ride video has no labels, so the reference is made by the detector itself with everything it has: every
second frame (10 Hz), three views at full detail (the wide band and two full-resolution tiles), no tracker.
Its boxes are linked over time; a physical light is a chain that lasts at least `--min-s` and is lit (red /
yellow / green by the state CNN) most of the time. What that reference cannot see either (a light too small
even at full resolution, glare) is missing from both sides: the numbers are "share of what the model can see
at its best", an upper bound of the true detection rate.

    lights_replay --no-track --thr 0.2 [--roi ...] FRAMES(every 2nd) > ref_*.jsonl      (one per view)
    lights_replay [production options] --dt-ms 350 FRAMES(every 7th) > variant.jsonl
    replay_score.py --ref ref_wide.jsonl:2 ref_zl.jsonl:2 ref_zr.jsonl:2 --run B=variant.jsonl:7 [--run ...]

(`:N` = the file holds every Nth frame of the 20 fps ride, starting with the first.) A consumer shows the
lights of the newest message until the next one, so a reference light counts as covered at a reference frame
when the newest message of the run (at most `--stale-s` old) lists a light on it.

Printed per run:
  lights      reference lights (chains) reported at least once / all
  time        share of the reference light-time covered, in all and by the housing height in pixels
  first       median delay from the first reference sighting to the first report, seconds
  drop-outs   covered -> not covered changes per minute of light-time (the blinking a consumer sees)
  extra       reported light-frames per minute with no reference box near (false or stale reports)
  msgs        share of the run's messages (while a reference light exists) that list at least one light
"""
import argparse
import json
import sys


MAX_FRAME = None


def load(spec):
    path, _, step = spec.rpartition(":")
    step = int(step)
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if MAX_FRAME is not None and i * step > MAX_FRAME:
                break
            if line.startswith("{"):
                try:
                    out.append((i * step, json.loads(line)))   # (ride frame index, message)
                except ValueError:                              # a file still being written
                    break
    return out


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def near(a, b, k=1.0):
    """Same light: overlapping, or centres closer than k x the larger long side (min 12 px)."""
    if iou(a, b) > 0.1:
        return True
    d = ((a[0] + a[2] - b[0] - b[2]) ** 2 + (a[1] + a[3] - b[1] - b[3]) ** 2) ** 0.5 / 2
    return d <= max(12.0, k * max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]))


def reference(specs, score, min_s, fps):
    """-> (chains, all_boxes): chains = [{"frames": {frame: box}, "lit": bool}], all_boxes = {frame: [box]}."""
    per = {}
    for spec in specs:
        for fr, m in load(spec):
            per.setdefault(fr, []).extend(m.get("raw") or [])
    chains, open_, all_boxes = [], [], {}
    for fr in sorted(per):
        dets = sorted(per[fr], key=lambda d: -d["score"])
        kept = []
        for d in dets:                                           # NMS across the views
            if all(iou(d["box"], k["box"]) < 0.45 for k in kept):
                kept.append(d)
        all_boxes[fr] = [d["box"] for d in kept]
        strong = [d for d in kept if d["score"] >= score]
        used = set()
        for ch in open_:
            best, bi = None, None
            for i, d in enumerate(strong):
                if i not in used and near(ch["last"], d["box"]):
                    u = iou(ch["last"], d["box"])
                    if best is None or u > best:
                        best, bi = u, i
            if bi is not None:
                used.add(bi)
                d = strong[bi]
                ch["frames"][fr] = d["box"]
                ch["states"].append(d.get("state"))
                ch["last"], ch["last_fr"] = d["box"], fr
        for i, d in enumerate(strong):
            if i not in used:
                open_.append({"frames": {fr: d["box"]}, "states": [d.get("state")], "last": d["box"], "last_fr": fr})
        keep = []
        for ch in open_:
            if fr - ch["last_fr"] > 0.6 * fps:                    # not seen for 0.6 s: the chain ends
                chains.append(ch)
            else:
                keep.append(ch)
        open_ = keep
    chains += open_
    out = []
    for ch in chains:
        frs = sorted(ch["frames"])
        if (frs[-1] - frs[0]) / fps < min_s or len(frs) < 5:
            continue
        lit = sum(1 for s in ch["states"] if s in ("red", "yellow", "green", "red_yellow"))
        ch["lit"] = lit >= 0.5 * len(ch["states"])
        # fill the gaps inside the chain (the light was there, the dense pass missed a frame)
        step = min(b - a for a, b in zip(frs, frs[1:]))
        full = {}
        for a, b in zip(frs, frs[1:]):
            n = (b - a) // step
            for k in range(n):
                t = k / n
                full[a + k * step] = [ch["frames"][a][j] * (1 - t) + ch["frames"][b][j] * t for j in range(4)]
        full[frs[-1]] = ch["frames"][frs[-1]]
        ch["frames"] = full
        out.append(ch)
    return out, all_boxes


def score_run(name, spec, chains, all_boxes, fps, stale_s):
    msgs = load(spec)
    idx = [fr for fr, _ in msgs]
    bins = [(0, 12), (12, 24), (24, 48), (48, 10 ** 6)]
    tot = {b: [0, 0] for b in bins}
    seen_chains, delays, drops, light_frames = 0, [], 0, 0
    import bisect
    for ch in chains:
        if not ch["lit"]:
            continue
        frs = sorted(ch["frames"])
        first, prev = None, None
        for fr in frs:
            k = bisect.bisect_right(idx, fr) - 1
            cov = False
            if k >= 0 and (fr - idx[k]) / fps <= stale_s:
                cov = any(near(ch["frames"][fr], li["box"]) for li in msgs[k][1].get("lights") or [])
            box = ch["frames"][fr]
            h = max(box[3] - box[1], box[2] - box[0])
            for b in bins:
                if b[0] <= h < b[1]:
                    tot[b][0] += cov
                    tot[b][1] += 1
            light_frames += 1
            if cov and first is None:
                first = fr
            if prev is True and not cov:
                drops += 1
            prev = cov
        if first is not None:
            seen_chains += 1
            delays.append((first - frs[0]) / fps)
    lit = [ch for ch in chains if ch["lit"]]
    ref_step = 2
    if lit:
        fr0 = sorted(lit[0]["frames"])
        ref_step = fr0[1] - fr0[0] if len(fr0) > 1 else 2
    light_min = light_frames * ref_step / fps / 60.0
    # extra: reported lights with no reference box (any score) within +-0.35 s
    ref_frames = sorted(all_boxes)
    extra = n_rep = with_light = msgs_during = 0
    lit_frames = set()
    for ch in lit:
        lit_frames.update(ch["frames"])
    for fr, m in msgs:
        ls = m.get("lights") or []
        a = bisect.bisect_left(ref_frames, fr - 0.35 * fps)
        b = bisect.bisect_right(ref_frames, fr + 0.35 * fps)
        boxes = [bx for f2 in ref_frames[a:b] for bx in all_boxes[f2]]
        for li in ls:
            n_rep += 1
            if not any(near(li["box"], bx) for bx in boxes):
                extra += 1
        if any(f2 in lit_frames for f2 in ref_frames[a:b]):
            msgs_during += 1
            with_light += bool(ls)
    run_min = (idx[-1] - idx[0]) / fps / 60.0 if len(idx) > 1 else 1.0
    cov_all = sum(v[0] for v in tot.values()) / max(1, sum(v[1] for v in tot.values()))
    delays.sort()
    print("%-4s lights %3d/%-3d  time %5.1f %%  [<12 px %s  12-24 %s  24-48 %s  >=48 %s]  first %.2f s  "
          "drop-outs %5.1f/min  extra %5.1f/min (%d of %d)  msgs %5.1f %%" % (
              name, seen_chains, len(lit), 100 * cov_all,
              *["%5.1f %% (%d)" % (100 * tot[b][0] / tot[b][1], tot[b][1]) if tot[b][1] else "   -   " for b in bins],
              delays[len(delays) // 2] if delays else float("nan"),
              drops / light_min if light_min else 0.0, extra / run_min, extra, n_rep,
              100.0 * with_light / max(1, msgs_during)))
    return cov_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", nargs="+", required=True, help="reference files FILE:STEP (lights_replay --no-track)")
    ap.add_argument("--run", action="append", default=[], help="NAME=FILE:STEP")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--score", type=float, default=0.35, help="reference boxes at or above this score form chains")
    ap.add_argument("--min-s", type=float, default=0.8)
    ap.add_argument("--stale-s", type=float, default=0.5)
    ap.add_argument("--max-frame", type=int, help="use the ride up to this frame index only")
    a = ap.parse_args()
    global MAX_FRAME
    MAX_FRAME = a.max_frame
    chains, all_boxes = reference(a.ref, a.score, a.min_s, a.fps)
    lit = [c for c in chains if c["lit"]]
    secs = sum(len(c["frames"]) for c in lit) * (sorted(all_boxes)[1] - sorted(all_boxes)[0]) / a.fps if len(all_boxes) > 1 else 0
    print("reference: %d frames, %d chains (%d lit, %.0f light-seconds), %d not lit (side / back views)" %
          (len(all_boxes), len(chains), len(lit), secs, len(chains) - len(lit)))
    for r in a.run:
        name, _, spec = r.partition("=")
        score_run(name, spec, chains, all_boxes, a.fps, a.stale_s)


if __name__ == "__main__":
    main()
