#!/usr/bin/env python3
"""car_state_cpp.py SCAN.npz SHEETS_DIR LIST.txt TAG=WIDE.jsonl,ZOOM.jsonl ... : the labelled car sample scored
on what signd's own pipeline says (lights_replay --no-track on the sample's frames, once per view; NV12
conversion, ncnn detector, state CNN as on the phone). A labelled detection takes the state of the raw box
with the best IoU (>= 0.4) in either view; detections the phone pipeline did not reproduce are counted apart.
LIST.txt names the frames in the order they were replayed (<session>_<seq>.jpg)."""
import json, os, sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=True); meta = json.loads(str(z["meta"]))
S = json.load(open(os.path.join(sys.argv[2], "sample.json")))
lab = dict((int(a), b) for a, b in (l.split() for l in open(os.path.join(sys.argv[2], "labels.txt"))))
files = [l.strip() for l in open(sys.argv[3]) if l.strip()]
n_none = S["strata"]["cnn_none"][1]
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)
MAP = {"red": "r", "yellow": "y", "green": "g", "red_yellow": "r", "off": "none", "unknown": "none"}
for spec in sys.argv[4:]:
    tag, paths = spec.split("=")
    raw = {}
    for p in paths.split(","):
        for f, line in zip(files, open(p)):
            raw.setdefault(f, []).extend(json.loads(line).get("raw") or [])
    lit = {"right": 0.0, "wrong": 0.0, "none": 0.0}; oth = {"lit": 0.0, "none": 0.0}; missing = 0
    for k, i in enumerate(S["sample"]):
        m = meta[i]; w = S["strata"]["cnn_none"][0] / n_none if k < n_none else S["strata"]["cnn_lit"][0] / (len(S["sample"]) - n_none)
        c = [(iou(m["box"], r["box"]), r) for r in raw.get("%s_%d.jpg" % (m["session"], m["seq"]), [])]
        c = [x for x in c if x[0] >= 0.4]
        if not c: missing += 1; continue
        p = MAP[max(c, key=lambda x: x[0])[1]["state"]]; l = lab[k]
        if l in ("r", "y", "g", "ry"):
            ok = p == l or (l == "ry" and p in ("r", "y"))
            lit["right" if ok else "none" if p == "none" else "wrong"] += w
        else: oth["lit" if p != "none" else "none"] += w
    tl, to = sum(lit.values()), sum(oth.values())
    print("%-6s lit lights: right %5.1f %%  wrong colour %4.1f %%  none %5.1f %%  | others called lit %5.1f %%  | %d of %d sample boxes not reproduced" % (
        tag, 100 * lit["right"] / tl, 100 * lit["wrong"] / tl, 100 * lit["none"] / tl, 100 * oth["lit"] / to, missing, len(S["sample"])))
