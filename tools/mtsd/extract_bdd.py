#!/usr/bin/env python3
"""Not-a-sign crops from BDD100K val (FiftyOne export: samples.json + data/) for the classifier's `background` class,
in the shard format of extract_crops.py (96x96 square region, box = 64 px on its long side).

usage: extract_bdd.py BDD_DIR L8_HARVEST_DIR DET.onnx OUT_DIR [--workers 7]

Per image (split by md5(image id): first hex digit >= c -> test, else train; the same split as roadsense L8):
  * bddlamp   L8's bright red blobs inside car/truck/bus boxes (stop/tail lamps; L8_HARVEST_DIR/bddblobs/blobs.json)
  * bddlight  L8's red traffic-light blobs (L8_HARVEST_DIR/bddlights/blobs.json)
  * lightgt   annotated traffic lights of any colour (>= 8 px, at most 4 per image)
  * detlight  detector boxes (phone view: 1280x720 letterboxed to 640x384, score > 0.2) on a traffic light (IoU >= 0.3)
  * detneg    detector boxes with IoU < 0.1 to every annotated traffic sign and centre outside them
All are labelled `background`. BDD100K is licensed for non-commercial research: shards are never committed.
"""
import argparse, hashlib, json, os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_crops as ec  # noqa: E402

G = {}


def init(bdd, det, blobs):
    import onnxruntime as ort
    cv2.setNumThreads(1)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    ec.G['sess'] = ort.InferenceSession(det, so, providers=['CPUExecutionProvider'])
    G['bdd'] = bdd; G['blobs'] = blobs


def work(s):
    path = G['bdd'] + '/' + s['filepath']; iid = os.path.basename(path)[:-4]
    split = 'test' if hashlib.md5(iid.encode()).hexdigest()[0] >= 'c' else 'train'
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None: return None, []
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB); H, W = im.shape[:2]
    tod = (s.get('timeofday') or {}).get('label')
    gts = (s.get('detections') or {}).get('detections') or []
    def px(g):
        x, y, w, h = g['bounding_box']; return [x * W, y * H, (x + w) * W, (y + h) * H]
    signs = np.array([px(g) for g in gts if g['label'] == 'traffic sign']).reshape(-1, 4)
    lights = np.array([px(g) for g in gts if g['label'] == 'traffic light']).reshape(-1, 4)
    X, M = [], []
    def add(b, kind):
        w, h = b[2] - b[0], b[3] - b[1]
        if w < 3 or h < 3: return
        c, _ = ec.crop(im, b); m = max(w, h)
        X.append(c); M.append({'key': iid, 'kind': kind, 'label': 'background', 'split': split, 'tod': tod,
                               'w': round(float(w), 1), 'h': round(float(h), 1), 'W': W, 'H': H,
                               'sw': round(float(64 * w / m), 2), 'sh': round(float(64 * h / m), 2)})
    for kind, lst in (('bddlamp', G['blobs'][0].get(iid, [])), ('bddlight', G['blobs'][1].get(iid, []))):
        for b in lst: add(np.array(b, float), kind)
    n = 0
    for b in lights:
        if n >= 4: break
        if min(b[2] - b[0], b[3] - b[1]) >= 8: add(b, 'lightgt'); n += 1
    D, Sc = ec.detect(im)
    Il, Is = ec.iou_mat(D, lights), ec.iou_mat(D, signs)
    for j in range(len(D)):
        if Il.shape[1] and Il[j].max() >= 0.3:
            add(D[j], 'detlight'); continue
        if Is.shape[1] and Is[j].max() >= 0.1: continue
        cx, cy = (D[j, 0] + D[j, 2]) / 2, (D[j, 1] + D[j, 3]) / 2
        if len(signs) and ((signs[:, 0] <= cx) & (cx <= signs[:, 2]) & (signs[:, 1] <= cy) & (cy <= signs[:, 3])).any(): continue
        add(D[j], 'detneg')
    return (np.stack(X) if X else None), M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bdd'); ap.add_argument('harvest'); ap.add_argument('det'); ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = json.load(open(a.bdd + '/samples.json'))['samples']
    blobs = []
    for f in ('bddblobs', 'bddlights'):
        d = {}
        for r in json.load(open(f'{a.harvest}/{f}/blobs.json')): d.setdefault(r['img'], []).append(r['box'])
        blobs.append(d)
    import multiprocessing as mp
    X, M = [], []
    with mp.Pool(a.workers, initializer=init, initargs=(a.bdd, a.det, blobs)) as pool:
        for i, (x, m) in enumerate(pool.imap_unordered(work, S, chunksize=16)):
            if x is not None: X.append(x); M += m
            if i % 2000 == 0: print(i, len(M), flush=True)
    np.savez_compressed(f'{a.out}/bdd_000.npz', X=np.concatenate(X), meta=json.dumps(M))
    import collections
    print(collections.Counter((m['kind'], m['split']) for m in M))


if __name__ == '__main__':
    main()
