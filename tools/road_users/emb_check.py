#!/usr/bin/env python3
"""emb_check.py OUT.jsonl: distances of the published `emb` (yuv3x3) between boxes of one object about 1 s apart
(chains by IoU at the unit's schedule) and between a box and the other boxes of the later pass."""
import json, sys, numpy as np
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)
def vec(u): return np.frombuffer(bytes.fromhex(u['emb']), np.int8).astype(np.float32) / 32
V = {'car', 'truck', 'bus', 'train'}
msgs = [json.loads(l) for l in open(sys.argv[1])]
chains = []
same, other = [], []
for k, m in enumerate(msgs):
    t = m['ts_mono_ns'] * 1e-9
    us = [u for u in (m.get('road_users') or []) if u['cls'] in V and 'emb' in u]
    used = set()
    for u in us:
        best, bs = None, 0.3
        for ci, c in enumerate(chains):
            if ci in used or t - c[-1][0] > 0.7 or c[-1][2] == k: continue
            s = iou(u['box'], c[-1][1]['box'])
            if s >= bs: best, bs = ci, s
        if best is None: chains.append([(t, u, k)]); c = chains[-1]
        else: chains[best].append((t, u, k)); used.add(best); c = chains[best]
        for (t0, u0, k0) in c[:-1]:
            if 0.75 <= t - t0 <= 1.25:
                same.append(float(np.linalg.norm(vec(u) - vec(u0))))
                for w in us:
                    if w is not u: other.append(float(np.linalg.norm(vec(w) - vec(u0))))
same, other = np.array(same), np.array(other)
q = [10, 25, 50, 75, 90]
print('%s: same object about 1 s apart: n %d, quantiles 10/25/50/75/90 %% %s; other boxes of the later pass: n %d, %s'
      % (sys.argv[1], len(same), np.percentile(same, q).round(2).tolist() if len(same) else [], len(other), np.percentile(other, q).round(2).tolist() if len(other) else []))
