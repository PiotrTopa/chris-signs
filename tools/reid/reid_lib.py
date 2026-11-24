"""Shared by tools/reid/*.py: ride crops (crops_extract.py), identities from dense IoU chains, the data-free
descriptors and the retrieval score. A ride file: rows [frame, class, score, x0, y0, x1, y1, overlap] and 32x32 BGR
crops (box + 5 % margin), every 2nd frame of a 20 fps ride."""
import numpy as np

def load(path):
    z = np.load(path)
    return z['rows'], z['crops']

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)

def group(cls):
    return np.where(np.isin(cls.astype(int), [2, 3, 4, 5]), 1, 0)

def chains(rows, step=2):
    """identity per box: a box continues a chain when it is the mutual best IoU >= 0.5 of the chain's last box (at
    most two processed frames back) and neither has a second candidate above 0.3; else it starts a chain"""
    fr = rows[:, 0].astype(int); box = rows[:, 3:7]; grp = group(rows[:, 1]); n = len(rows)
    byf = {}
    for i in range(n): byf.setdefault(fr[i], []).append(i)
    tid = -np.ones(n, int); nt = 0; last = {}
    for f in sorted(byf):
        cur = byf[f]
        cands = {t: v for t, v in last.items() if f - v[0] <= 2 * step}
        for t in [t for t in last if t not in cands]: del last[t]
        pairs = []
        for i in cur:
            for t, (pf, j) in cands.items():
                if grp[i] != grp[j]: continue
                u = iou(box[i], box[j])
                if u > 0.3: pairs.append((u, i, t))
        pairs.sort(reverse=True)
        ci, ct = {}, {}
        for u, i, t in pairs: ci[i] = ci.get(i, 0) + 1; ct[t] = ct.get(t, 0) + 1
        di, dt = set(), set()
        for u, i, t in pairs:
            if i in di or t in dt: continue
            di.add(i); dt.add(t)
            if u >= 0.5 and ci[i] == 1 and ct[t] == 1: tid[i] = t
        for i in cur:
            if tid[i] < 0: tid[i] = nt; nt += 1
            last[tid[i]] = (f, i)
    return tid

def yuv(crops):
    c = crops.astype(np.float32); b, g, r = c[..., 0], c[..., 1], c[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y, 128 + 0.564 * (b - y), 128 + 0.713 * (r - y)

def inner(a, m=3):  # the central 26 of 32 pixels: about the inner 80 % of the box
    return a[:, m:32 - m, m:32 - m]

def cells(a, g):
    n, s = a.shape[0], a.shape[1]
    e = np.linspace(0, s, g + 1).round().astype(int)
    return np.stack([a[:, e[i]:e[i + 1], e[j]:e[j + 1]].mean((1, 2)) for i in range(g) for j in range(g)], 1)

def desc_yuv_grid(crops, g=3, wy=1.0, wc=1.0, demean_y=False):
    """g x g cells of the inner box: mean Y, Cb, Cr; Y in units of 50 grey levels, chroma in units of 20"""
    y, cb, cr = [inner(a) for a in yuv(crops)]
    Y = (cells(y, g) - 110) / 50
    if demean_y:
        Y = Y - Y.mean(1, keepdims=True)
    return np.concatenate([wy * Y, wc * (cells(cb, g) - 128) / 20, wc * (cells(cr, g) - 128) / 20], 1)

def retrieval(X, rows, tid, gap_s, fps=20.0, step=2, sub=2, grp_sel=1, minw=0, maxw=1e9):
    """query at frame t, its own box gap_s later against every box (of the subsampled frames) of every other chain
    of the ride: rank-1 share, mean AUC; frames subsampled to every `sub`-th processed frame (the unit's 5 Hz)"""
    fr = rows[:, 0].astype(int); w = rows[:, 5] - rows[:, 3]; grp = group(rows[:, 1])
    selm = (fr % (step * sub) == 0) & (grp == grp_sel)
    idx = np.where(selm)[0]
    G = X[idx]; gt = tid[idx]
    byf = {}
    for i in idx: byf.setdefault(fr[i], []).append(i)
    gap = int(round(gap_s * fps))
    tot = r1 = 0; aucs = []
    for f, qs in byf.items():
        gal = byf.get(f + gap)
        if not gal: continue
        for q in qs:
            if not (minw <= w[q] < maxw): continue
            same = [g for g in gal if tid[g] == tid[q]]
            if not same: continue
            d = np.linalg.norm(G - X[q], axis=1)
            dpos = np.linalg.norm(X[same[0]] - X[q])
            neg = gt != tid[q]
            rank = int((d[neg] < dpos).sum())
            tot += 1; r1 += rank == 0; aucs.append(1 - rank / max(1, neg.sum()))
    return (tot, r1 / max(1, tot), float(np.mean(aucs)) if aucs else 0.0)

def retrieval_frame(X, rows, tid, gap_s, fps=20.0, step=2, sub=2, grp_sel=1, min_other=1):
    """as retrieval(), against the other boxes of the SAME FRAME as the query's own later box (at least min_other
    of them): the association a tracker faces, and free of the chains' fragments (two boxes of one frame are two
    objects). Returns (queries, rank-1 share, mean AUC, mean number of other boxes)."""
    fr = rows[:, 0].astype(int); grp = group(rows[:, 1])
    idx = np.where((fr % (step * sub) == 0) & (grp == grp_sel))[0]
    byf = {}
    for i in idx: byf.setdefault(fr[i], []).append(i)
    gap = int(round(gap_s * fps)); tot = r1 = 0; aucs = []; nother = []
    for f, qs in byf.items():
        gal = byf.get(f + gap)
        if not gal or len(gal) < min_other + 1: continue
        G = X[gal]
        for q in qs:
            same = [k for k, g in enumerate(gal) if tid[g] == tid[q]]
            if not same: continue
            d = np.linalg.norm(G - X[q], axis=1)
            rank = int((np.delete(d, same[0]) < d[same[0]]).sum())
            tot += 1; r1 += rank == 0; aucs.append(1 - rank / (len(gal) - 1)); nother.append(len(gal) - 1)
    return (tot, r1 / max(1, tot), float(np.mean(aucs)) if aucs else 0.0, float(np.mean(nother)) if nother else 0.0)

def pair_distances(X, rows, tid, gap_s, fps=20.0, step=2, sub=2, grp_sel=1, same_frame_only=True):
    """distances of same-object pairs gap_s apart and of different-object pairs (query at t, the other boxes of the
    frame gap_s later): what a tracker's association compares"""
    fr = rows[:, 0].astype(int); grp = group(rows[:, 1])
    idx = np.where((fr % (step * sub) == 0) & (grp == grp_sel))[0]
    byf = {}
    for i in idx: byf.setdefault(fr[i], []).append(i)
    gap = int(round(gap_s * fps)); pos, neg = [], []
    for f, qs in byf.items():
        gal = byf.get(f + gap)
        if not gal: continue
        for q in qs:
            for g in gal:
                d = float(np.linalg.norm(X[g] - X[q]))
                (pos if tid[g] == tid[q] else neg).append(d)
    return np.array(pos), np.array(neg)
