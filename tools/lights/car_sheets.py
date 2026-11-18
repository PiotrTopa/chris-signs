#!/usr/bin/env python3
"""car_sheets.py SCAN.npz OUTDIR [--n 360] [--seed 1] [--min-score 0.25] [--skip SESSION,...] : contact sheets of detections from
car_scan.py for labelling by eye. Half of the sample are detections the state CNN calls `none`, half the
rest (stratified; sample.json keeps the strata sizes for weighting). Every cell: the crop with half a box
of context on each side, enlarged; under it the sample index. Labels go into OUTDIR/labels.txt as
`<index> <r|y|g|ry|o|x>` (o = a facing light that is dark, x = not a facing traffic light / unreadable)."""
import json, os, sys
import numpy as np, cv2
a = sys.argv
n = int(a[a.index("--n") + 1]) if "--n" in a else 360
seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 1
ms = float(a[a.index("--min-score") + 1]) if "--min-score" in a else 0.25
z = np.load(a[1], allow_pickle=True); meta = json.loads(str(z["meta"])); crops = z["crops"]
os.makedirs(a[2], exist_ok=True)
skip = a[a.index("--skip") + 1].split(",") if "--skip" in a else []      # sessions to leave out (e.g. parked)
idx = [i for i, m in enumerate(meta) if m["score"] >= ms and m["session"] not in skip]
off = [i for i in idx if int(np.argmax(meta[i]["cnn"])) == 3]; on = [i for i in idx if int(np.argmax(meta[i]["cnn"])) != 3]
rng = np.random.default_rng(seed)
so = sorted(rng.choice(off, min(n // 2, len(off)), replace=False).tolist()); sn = sorted(rng.choice(on, min(n - len(so), len(on)), replace=False).tolist())
sample = so + sn
json.dump({"scan": a[1], "min_score": ms, "strata": {"cnn_none": [len(off), len(so)], "cnn_lit": [len(on), len(sn)]}, "sample": sample}, open(os.path.join(a[2], "sample.json"), "w"))
CW, CH, COLS, ROWS = 150, 230, 8, 5
for s in range(0, len(sample), COLS * ROWS):
    sheet = np.full((ROWS * CH, COLS * CW, 3), 40, np.uint8)
    for k, i in enumerate(sample[s:s + COLS * ROWS]):
        c = crops[i]; h, w = c.shape[:2]; f = min((CW - 8) / w, (CH - 30) / h)
        im = cv2.resize(c, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_CUBIC)
        y0, x0 = (k // COLS) * CH + 2, (k % COLS) * CW + 4
        sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
        bh = meta[i]["box"][3] - meta[i]["box"][1]
        cv2.putText(sheet, "%d  %dpx" % (s + k, bh), (x0, y0 + CH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(a[2], "sheet%02d.png" % (s // (COLS * ROWS))), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
print("sample", len(sample), "strata", len(off), len(on))
