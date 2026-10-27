#!/usr/bin/env python3
"""Crops for the sign classifier from MTSD image zips (streamed, nothing unpacked), with hard negatives mined by
the current detector.

usage: extract_crops.py --ann ANN_DIR --det DET.onnx --out OUT_DIR [--workers 7] ZIP [ZIP ...]

ANN_DIR is mtsd_v2_fully_annotated/ (annotations/*.json). For every non-panorama image:
  * gt     every annotated object (not `dummy`), label = MTSD label (`other-sign` included)
  * detpos detections of the given detector matched to an annotation (IoU >= 0.5): the detector's own box with the
           annotation's label (box noise as on the phone)
  * detneg detections with IoU < 0.1 to every annotation and their centre outside every annotation box: label
           `background` (the detector's false positives on MTSD)
  * rand   2 random boxes of sign-like size away from every annotation: label `background`
The detector runs on three views: the whole image letterboxed to 640x384, and the image scaled to 1280 px wide with
two 1280x720 windows (rows 0-720 and the centre) letterboxed like a phone frame (scale 0.5).
Each crop is the square region of side 1.5 * max(box w, h) centred on the box, resized to 96x96 (edge-replicated
outside the image); the box then spans 64 px on its long side. Shards: OUT/<zip>_<k>.npz with X uint8 [n,96,96,3]
and a JSON list of metadata (image key, kind, label, flags, box w/h in image px, image w/h, det score, box in stored
px). MTSD is CC BY-NC-SA 4.0: shards are derived data, never committed.
"""
import argparse, json, os, sys, zipfile, hashlib, random
import numpy as np
import cv2

S, BOX = 96, 64
G = {}


def init(zpath, ann, det, threads):
    import onnxruntime as ort
    cv2.setNumThreads(1)
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    G['z'] = zipfile.ZipFile(zpath)
    G['ann'] = ann
    G['sess'] = ort.InferenceSession(det, so, providers=['CPUExecutionProvider'])


def letterbox(im, W=640, H=384):
    h0, w0 = im.shape[:2]
    r = min(W / w0, H / h0)
    nw, nh = round(w0 * r), round(h0 * r)
    left, top = round((W - nw) / 2 - 0.1), round((H - nh) / 2 - 0.1)
    can = np.full((H, W, 3), 114, np.uint8)
    can[top:top + nh, left:left + nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return can, r, left, top


def nms(b, s, thr=0.45):
    order = np.argsort(-s); keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1: break
        r = order[1:]
        xx0 = np.maximum(b[i, 0], b[r, 0]); yy0 = np.maximum(b[i, 1], b[r, 1])
        xx1 = np.minimum(b[i, 2], b[r, 2]); yy1 = np.minimum(b[i, 3], b[r, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        a = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1]); ar = (b[r, 2] - b[r, 0]) * (b[r, 3] - b[r, 1])
        order = r[inter / (a + ar - inter + 1e-9) <= thr]
    return keep


def detect(im, thr=0.2):
    can, r, left, top = letterbox(im)
    x = can.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    o = G['sess'].run(None, {'images': x})[0][0]   # [5, 5040]
    m = o[4] > thr
    if not m.any(): return np.zeros((0, 4)), np.zeros(0)
    cx, cy, w, h, s = o[:, m]
    b = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    k = nms(b, s)
    b, s = b[k], s[k]
    b[:, [0, 2]] = (b[:, [0, 2]] - left) / r; b[:, [1, 3]] = (b[:, [1, 3]] - top) / r
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, im.shape[1]); b[:, [1, 3]] = b[:, [1, 3]].clip(0, im.shape[0])
    ok = ((b[:, 2] - b[:, 0]) >= 4) & ((b[:, 3] - b[:, 1]) >= 4)
    return b[ok], s[ok]


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)))
    xx0 = np.maximum(a[:, None, 0], b[None, :, 0]); yy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx1 = np.minimum(a[:, None, 2], b[None, :, 2]); yy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def crop(im, b):
    x0, y0, x1, y1 = b
    w, h = x1 - x0, y1 - y0
    side = 1.5 * max(w, h); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sc = S / side
    # affine: out = (in - (c - side/2)) * sc
    M = np.array([[sc, 0, -(cx - side / 2) * sc], [0, sc, -(cy - side / 2) * sc]], np.float32)
    interp = cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR
    if sc < 1:   # pre-shrink for area-like quality, warpAffine has no true INTER_AREA
        f = max(1, int(1 / sc))
        if f > 1:
            X0, Y0 = max(0, int(cx - side / 2) - f), max(0, int(cy - side / 2) - f)
            X1, Y1 = min(im.shape[1], int(cx + side / 2) + f + 1), min(im.shape[0], int(cy + side / 2) + f + 1)
            sub = im[Y0:Y1, X0:X1]
            sub = cv2.resize(sub, (max(1, (X1 - X0) // f), max(1, (Y1 - Y0) // f)), interpolation=cv2.INTER_AREA)
            fx, fy = (X1 - X0) / sub.shape[1], (Y1 - Y0) / sub.shape[0]
            M = np.array([[sc * fx, 0, (X0 - (cx - side / 2)) * sc], [0, sc * fy, (Y0 - (cy - side / 2)) * sc]], np.float32)
            return cv2.warpAffine(sub, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), sc
    return cv2.warpAffine(im, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), sc


def work(name):
    key = os.path.basename(name)[:-4]
    try:
        a = json.load(open(os.path.join(G['ann'], 'annotations', key + '.json')))
    except FileNotFoundError:
        return key, None, None
    if a.get('ispano'):
        return key, None, None
    im = cv2.imdecode(np.frombuffer(G['z'].read(name), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        return key, None, None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    H, W = im.shape[:2]
    objs = [o for o in a['objects'] if 'cross_boundary' not in o['bbox']]
    gtb = np.array([[o['bbox']['xmin'], o['bbox']['ymin'], o['bbox']['xmax'], o['bbox']['ymax']] for o in objs]).reshape(-1, 4)
    gtb[:, [0, 2]] *= W / a['width']; gtb[:, [1, 3]] *= H / a['height']
    X, M = [], []

    def add(b, kind, label, flags=None, score=None, gi=None):
        w, h = b[2] - b[0], b[3] - b[1]
        if w < 2 or h < 2: return
        c, sc = crop(im, b)
        m = max(w, h)
        X.append(c)
        M.append({'key': key, 'kind': kind, 'label': label, 'flags': flags or [], 'w': round(float(w), 1),
                  'h': round(float(h), 1), 'W': W, 'H': H, 'score': None if score is None else round(float(score), 3),
                  'sw': round(float(BOX * w / m), 2), 'sh': round(float(BOX * h / m), 2), 'gi': gi})

    for i, o in enumerate(objs):
        fl = [k for k, v in o['properties'].items() if v]
        if 'dummy' in fl: continue
        add(gtb[i], 'gt', o['label'], fl, gi=i)
    # detector views
    dets, scs, views = [], [], []
    b, s = detect(im); dets.append(b); scs.append(s); views += [0] * len(s)
    r = 1280 / W
    sm = cv2.resize(im, (1280, max(1, round(H * r))), interpolation=cv2.INTER_AREA)
    for vi, y0 in enumerate(sorted({0, max(0, (sm.shape[0] - 720) // 2)})):
        win = sm[y0:y0 + 720]
        b, s = detect(win)
        b = b.copy(); b[:, [1, 3]] += y0; b /= r
        dets.append(b); scs.append(s); views += [vi + 1] * len(s)
    D = np.concatenate(dets); Sc = np.concatenate(scs)
    if len(D):
        k = nms(D, Sc, 0.6); D, Sc = D[k], Sc[k]
    I = iou_mat(D, gtb)
    for j in range(len(D)):
        best = I[j].max() if I.shape[1] else 0.0
        if best >= 0.5:
            gi = int(I[j].argmax()); o = objs[gi]
            fl = [k for k, v in o['properties'].items() if v]
            if 'dummy' in fl: continue
            add(D[j], 'detpos', o['label'], fl, Sc[j], gi)
        elif best < 0.1:
            cx, cy = (D[j, 0] + D[j, 2]) / 2, (D[j, 1] + D[j, 3]) / 2
            mw = 0.1 * (gtb[:, 2] - gtb[:, 0]); mh = 0.1 * (gtb[:, 3] - gtb[:, 1])
            inside = ((gtb[:, 0] - mw <= cx) & (cx <= gtb[:, 2] + mw) & (gtb[:, 1] - mh <= cy) & (cy <= gtb[:, 3] + mh)).any() if len(gtb) else False
            if not inside:
                add(D[j], 'detneg', 'background', [], Sc[j])
    rng = random.Random(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))
    got = 0
    for _ in range(30):
        if got >= 2: break
        side = W * np.exp(rng.uniform(np.log(0.006), np.log(0.06)))
        asp = rng.choice([1.0, 1.0, 1.0, 0.7, 1.4])
        w, h = side * min(1, asp), side * min(1, 1 / asp)
        x0, y0 = rng.uniform(0, W - w), rng.uniform(0, H - h)
        b = np.array([[x0, y0, x0 + w, y0 + h]])
        big = gtb.copy()
        if len(big):
            ew = 0.5 * (big[:, 2] - big[:, 0]); eh = 0.5 * (big[:, 3] - big[:, 1])
            big[:, 0] -= ew; big[:, 2] += ew; big[:, 1] -= eh; big[:, 3] += eh
            if iou_mat(b, big).max() > 0: continue
        add(b[0], 'rand', 'background')
        got += 1
    return key, (np.stack(X) if X else np.zeros((0, S, S, 3), np.uint8)), M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True); ap.add_argument('--det', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=7); ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--shard', type=int, default=1500)
    ap.add_argument('zips', nargs='+')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import multiprocessing as mp
    for zp in a.zips:
        tag = os.path.basename(zp)[:-4]
        names = [n for n in zipfile.ZipFile(zp).namelist() if n.endswith('.jpg')]
        if a.limit: names = names[:a.limit]
        done = os.path.join(a.out, tag + '.done')
        if os.path.exists(done):
            print(tag, 'already done', flush=True); continue
        print(tag, len(names), 'images', flush=True)
        X, M, k, n = [], [], 0, 0
        with mp.Pool(a.workers, initializer=init, initargs=(zp, a.ann, a.det, a.threads)) as pool:
            for key, x, m in pool.imap_unordered(work, names, chunksize=8):
                n += 1
                if x is not None and len(x):
                    X.append(x); M += m
                if sum(len(t) for t in X) >= a.shard * 10 or (n == len(names) and X):
                    np.savez_compressed(os.path.join(a.out, f'{tag}_{k:03d}.npz'), X=np.concatenate(X), meta=json.dumps(M))
                    k += 1; X, M = [], []
                if n % 1000 == 0: print(tag, n, flush=True)
        open(done, 'w').write(str(n))
    print('done', flush=True)


if __name__ == '__main__':
    main()
