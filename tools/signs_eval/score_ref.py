#!/usr/bin/env python3
"""Score sign variants against the "detector at its best" reference (lane L24-S).

Reference: the whole-frame view and three full-scale tiles around the horizon (cache_dets.py A + T3) on every
second frame (10 Hz), linked by signd's own tracker (signs_track_replay, min_hits 1); a reference sign is a track
of >= 4 hits whose class is a sign (not `background`, not suppressed). Like the L15 light reference it can only
contain what the detector sees at full scale: shares are upper bounds of the true rates.

  score_ref.py SLUG REF.jsonl NAME=VARIANT.jsonl ... [--json OUT]

Per variant:
  time  share of the reference sign-time (reference frames x signs) at which the newest variant message (at most
        0.35 s old) lists a non-suppressed track near the reference box, by box size (max side, px) bands
  found reference signs listed at least once, by the sign's largest size band
  lead  median time from the first listing to the reference sign's last sighting (~ passing), on signs found by the
        first variant and this one (paired)
  extra ended non-suppressed sign tracks of the variant with no reference box (any score >= 0.2) near any of their
        boxes, per minute
"""
import argparse
import json
import os
import sys

import numpy as np

H = os.path.expanduser("~")
BANDS = [(0, 12, "<12"), (12, 16, "12-16"), (16, 24, "16-24"), (24, 1e9, "24+")]


def band(sz):
    for a, b, n in BANDS:
        if a <= sz < b:
            return n


def near(a, b, k=1.0):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix > 0 and iy > 0:
        inter = ix * iy
        if inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter) > 0.1:
            return True
    d = ((a[0] + a[2] - b[0] - b[2]) ** 2 + (a[1] + a[3] - b[1] - b[3]) ** 2) ** 0.5 / 2
    return d <= max(12.0, k * max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]))


def load_run(path):
    msgs = [json.loads(l) for l in open(path)]
    per_frame = {}      # frame -> [(track_id, box, suppressed)]
    tracks = {}         # track_id -> {"frames": {frame: box}}
    ended = {}
    for m in msgs:
        f = m["frame"]
        per_frame[f] = [(d["track_id"], d["box"], bool(d.get("suppressed"))) for d in m.get("detections") or []]
        for d in m.get("detections") or []:
            tracks.setdefault(d["track_id"], {})[f] = d["box"]
        for e in m.get("tracks_ended") or []:
            ended[e["track_id"]] = e
    return per_frame, tracks, ended


def is_sign(e):
    return not e.get("suppressed") and e.get("cls") != "background"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("ref")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--stale", type=int, default=7, help="frames (20 fps)")
    ap.add_argument("--cache", default=f"{H}/datasets/l24s/cache")
    a = ap.parse_args()
    rf, rtracks, rended = load_run(a.ref)
    refs = {}
    for tid, fr in rtracks.items():
        e = rended.get(tid)
        if e is None or not is_sign(e) or e["hits"] < 4:
            continue
        refs[tid] = fr
    # all reference boxes (score >= 0.2) for the "extra" test
    allref = {}
    for nm in ("A", "T3"):
        z = np.load(f"{a.cache}/{a.slug}.{nm}.npz")
        for f, b in zip(z["frame"], z["box"]):
            if f % 2 == 0:
                allref.setdefault(int(f), []).append(b.tolist())
    ref_frames = sorted(rf)
    minutes = (ref_frames[-1] - ref_frames[0]) / 20 / 60
    out = {"slug": a.slug, "ref_signs": len(refs), "minutes": minutes, "runs": {}}
    first = None
    for spec in a.runs:
        name, path = spec.split("=", 1)
        pf, vt, ve = load_run(path)
        vframes = np.array(sorted(pf))
        tot = {n: 0 for *_, n in BANDS}
        cov = {n: 0 for *_, n in BANDS}
        found_b = {n: [0, 0] for *_, n in BANDS}
        first_cov = {}
        for tid, fr in refs.items():
            mx = max(max(b[2] - b[0], b[3] - b[1]) for b in fr.values())
            bb = band(mx)
            found_b[bb][1] += 1
            for f in sorted(fr):
                box = fr[f]
                n = band(max(box[2] - box[0], box[3] - box[1]))
                tot[n] += 1
                i = np.searchsorted(vframes, f, side="right") - 1
                ok = False
                if i >= 0 and f - vframes[i] <= a.stale:
                    for vid, vb, sup in pf[int(vframes[i])]:
                        if not sup and near(box, vb) and is_sign(ve.get(vid, {"cls": "?"})):
                            ok = True
                            break
                if ok:
                    cov[n] += 1
                    first_cov.setdefault(tid, f)
            if tid in first_cov:
                found_b[bb][0] += 1
        lead = {tid: (max(refs[tid]) - first_cov[tid]) / 20.0 for tid in first_cov}
        # extra tracks
        n_extra, n_tr = 0, 0
        for vid, e in ve.items():
            if not is_sign(e) or vid not in vt:
                continue
            n_tr += 1
            hit = False
            for f, vb in vt[vid].items():
                g = f - (f % 2)
                for gg in (g, g + 2, g - 2):
                    if any(near(vb, rb) for rb in allref.get(gg, [])):
                        hit = True
                        break
                if hit:
                    break
            n_extra += not hit
        r = {"time": {n: (cov[n] / tot[n] if tot[n] else None) for n in tot}, "time_n": tot,
             "time_all": sum(cov.values()) / max(1, sum(tot.values())),
             "found": {n: found_b[n] for n in found_b}, "found_all": [len(first_cov), len(refs)],
             "lead": lead, "tracks": n_tr, "extra": n_extra, "extra_per_min": n_extra / minutes}
        if first is None:
            first = lead
        paired = [k for k in lead if k in first]
        r["lead_paired_median"] = float(np.median([lead[k] for k in paired])) if paired else None
        r["lead_base_paired_median"] = float(np.median([first[k] for k in paired])) if paired else None
        r["lead_n_paired"] = len(paired)
        out["runs"][name] = r
        t = r["time"]
        print(f"{a.slug} {name:14s} time all {r['time_all']:.3f} | " +
              " ".join(f"{n} {t[n]:.3f}" if t[n] is not None else f"{n} -" for n in t) +
              f" | found {len(first_cov)}/{len(refs)} " +
              " ".join(f"{n} {found_b[n][0]}/{found_b[n][1]}" for n in found_b) +
              f" | lead {r['lead_paired_median']} (base {r['lead_base_paired_median']}, n {len(paired)})"
              f" | tracks {n_tr} extra {n_extra} ({r['extra_per_min']:.2f}/min)")
    if a.json:
        json.dump(out, open(a.json, "w"))


if __name__ == "__main__":
    main()
