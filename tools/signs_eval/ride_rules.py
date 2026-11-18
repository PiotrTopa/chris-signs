#!/usr/bin/env python3
"""Put a variant's `signs` messages (variant.py output, with "frame") into an L11 phone recording of the ride in
place of the recorded ones, right after the camera message of that ride frame and with its seq / timestamps,
annotated by the api's signctx like the live api does. The result feeds roadsense tools/rules_replay.py.
(As roadsense docs/lanes/L12-mtsd-retrain/ride_signs.py, but for any processed-frame schedule.)

  ride_rules.py SLUG REC.jsonl VARIANT.jsonl OUT.jsonl [--roadsense DIR]
"""
import json
import os
import sys

H = os.path.expanduser("~")
args = sys.argv[1:]
rs = f"{H}/projects/wt/roadsense-v2"
if "--roadsense" in args:
    k = args.index("--roadsense")
    rs = args[k + 1]
    del args[k:k + 2]
slug, rec, var, out = args
sys.path.insert(0, f"{rs}/api")
from roadsense_api import signctx  # noqa: E402

vmsgs = {}
seq_of = {}
for l in open(var):
    m = json.loads(l)
    vmsgs.setdefault(m["frame"], []).append(m)   # the final flush shares the last frame
cr = json.load(open(f"{H}/datasets/rides/{slug}/camera-replay.json"))
try:
    calib = json.load(open(f"{H}/datasets/rides/{slug}/calib.json"))
except FileNotFoundError:
    calib = None
intr = (cr.get("focal_px", 690.0), cr.get("cx", 640.0), cr.get("cy", 360.0), 1280, 720)
ctx = signctx.SignContext()
latest = {}
frame_seq = {}
pending = []
with open(out, "w") as fo:
    for l in open(rec):
        w = json.loads(l)
        m = w["m"]
        t = m.get("type")
        if t == "signs":
            continue
        fo.write(json.dumps(w, separators=(",", ":")) + "\n")
        if t in ("road", "lights"):
            latest[t] = m
        if t == "camera" and m.get("file_frame") in vmsgs:
            frame_seq_map = {}
            for n in vmsgs.pop(m["file_frame"]):
                vs = n["seq"]
                n = dict(n)
                n.pop("frame", None)
                n.update(seq=m["seq"], ts_mono_ns=m["ts_mono_ns"], ts_real_ns=m.get("ts_real_ns"))
                frame_seq[vs] = m["seq"]
                for e in n.get("tracks_ended") or []:
                    for f in ("first_seq", "last_seq"):
                        if e.get(f) in frame_seq:
                            e[f] = frame_seq[e[f]]
                ctx.annotate(n, latest.get("road"), latest.get("lights"), calib, intr)
                fo.write(json.dumps({"t": w["t"], "m": n}, separators=(",", ":")) + "\n")
print(f"{slug}: {len(frame_seq)} signs messages placed, {sum(len(v) for v in vmsgs.values())} without a camera frame",
      file=sys.stderr)
