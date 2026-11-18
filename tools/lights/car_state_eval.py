#!/usr/bin/env python3
"""car_state_eval.py SCAN.npz SHEETS_DIR [STATE.onnx ...] : score state classifiers on the labelled sample of
detections from the car's own camera (car_scan.py + car_sheets.py + labels.txt). Per model: of the lights
labelled lit (r / y / g / ry), the share given the right colour, a wrong colour, `none`; of the rest (not a
facing light, dark, unreadable), the share called lit. Weighted by the strata of the sample, so the figures
estimate the whole set of detections. `rule` = the colour rule of signd/lights.cc alone; `cnn+rule` etc. are
not models but decision variants (see the code)."""
import json, os, sys
import numpy as np, cv2, onnxruntime as ort
z = np.load(sys.argv[1], allow_pickle=True); meta = json.loads(str(z["meta"])); crops = z["crops"]
sd = sys.argv[2]; S = json.load(open(os.path.join(sd, "sample.json")))
lab = dict((int(a), b) for a, b in (l.split() for l in open(os.path.join(sd, "labels.txt"))))
sample = S["sample"]; n_none = S["strata"]["cnn_none"][1]
wt = [S["strata"]["cnn_none"][0] / S["strata"]["cnn_none"][1]] * n_none + [S["strata"]["cnn_lit"][0] / max(1, S["strata"]["cnn_lit"][1])] * (len(sample) - n_none)
NAMES = ["r", "y", "g", "none"]

def box_crop(i, scale=1.0):
    m = meta[i]; c = crops[i]; X0, Y0 = m["ctx"][0], m["ctx"][1]; b = m["box"]
    cx, cy, w, h = (b[0] + b[2]) / 2 - X0, (b[1] + b[3]) / 2 - Y0, (b[2] - b[0]) * scale, (b[3] - b[1]) * scale
    x0, y0 = int(max(0, round(cx - w / 2))), int(max(0, round(cy - h / 2)))
    cc = c[y0:max(y0 + 3, int(round(cy + h / 2))), x0:max(x0 + 2, int(round(cx + w / 2)))]
    if m["horizontal"]: cc = cv2.rotate(cc, cv2.ROTATE_90_CLOCKWISE)
    return cc

def cnn(sess, i, scale=1.0):
    x = cv2.resize(box_crop(i, scale), (24, 48), interpolation=cv2.INTER_LINEAR).astype(np.float32).transpose(2, 0, 1)[None] / 255
    lg = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]; p = np.exp(lg - lg.max()); return p / p.sum()

def score(name, pred):
    lit = {"right": 0.0, "wrong": 0.0, "none": 0.0}; oth = {"lit": 0.0, "none": 0.0}; by = {}
    for k, i in enumerate(sample):
        l = lab[k]; p = pred(i); w = wt[k]
        if l in ("r", "y", "g", "ry"):
            ok = (p == l) or (l == "ry" and p in ("r", "y"))
            key = "right" if ok else "none" if p == "none" else "wrong"
            lit[key] += w; e = by.setdefault(l, [0.0, 0.0]); e[0] += w * ok; e[1] += w
        else:
            oth["lit" if p != "none" else "none"] += w
    tl = sum(lit.values()); to = sum(oth.values())
    print("%-28s lit lights: right %5.1f %%  wrong colour %4.1f %%  none %5.1f %%   | others called lit %5.1f %%   | recall %s" % (
        name, 100 * lit["right"] / tl, 100 * lit["wrong"] / tl, 100 * lit["none"] / tl, 100 * oth["lit"] / to,
        " ".join("%s %.0f%%" % (c, 100 * v[0] / v[1]) for c, v in sorted(by.items()))))

print("sample %d (weights: cnn-none x%.2f, cnn-lit x%.2f)" % (len(sample), wt[0], wt[-1]))
for path in sys.argv[3:]:
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    P = {i: cnn(s, i) for i in sample}
    score(os.path.basename(path), lambda i: NAMES[int(np.argmax(P[i]))])
    for thr in (0.6, 0.75, 0.9):       # `none` only when the model is sure of it, else the best lit class
        score("  none needs p>=%.2f" % thr, lambda i: "none" if P[i][3] >= thr else NAMES[int(np.argmax(P[i][:3]))])
