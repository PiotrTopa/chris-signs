#!/usr/bin/env python3
"""Hard negatives from BDD100K val without the detector: bright red blobs inside car/truck/bus boxes (tail/brake lamps)
and red traffic lights (GT colour 'red'). Saves square crops (blob box, the verifier's input region) with 15 % margin
like the harvests. usage: mine_bdd_blobs.py BDD_DIR OUTDIR [max_per_image]"""
import sys, os, json, hashlib, numpy as np
from PIL import Image
bdd, out = sys.argv[1], sys.argv[2]; mpi = int(sys.argv[3]) if len(sys.argv) > 3 else 4
LIGHTS_ONLY = len(sys.argv) > 4 and sys.argv[4] == 'lights'
os.makedirs(out + '/crops', exist_ok=True)
S = json.load(open(bdd + '/samples.json'))['samples']
def blobs(a):
    """connected components (4-neighbour, scipy-free) of bright red pixels in a region; returns boxes"""
    R, G, B = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    m = (R >= 180) & (R - G >= 50) & (R - B >= 40) | ((np.minimum(np.minimum(R, G), B) >= 230) & (R >= 240))
    h, w = m.shape; lab = np.zeros((h, w), np.int32); n = 0; res = []
    for y0, x0 in zip(*np.nonzero(m)):
        if lab[y0, x0]: continue
        n += 1; st = [(y0, x0)]; lab[y0, x0] = n; ys = []; xs = []
        while st:
            y, x = st.pop(); ys.append(y); xs.append(x)
            for yy, xx in ((y+1, x), (y-1, x), (y, x+1), (y, x-1)):
                if 0 <= yy < h and 0 <= xx < w and m[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n; st.append((yy, xx))
        if len(ys) >= 12: res.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1, len(ys)))
    return res
rows = []
for si, s in enumerate(S):
    path = bdd + '/' + s['filepath']; iid = os.path.basename(path)[:-4]
    split = 'test' if hashlib.md5(iid.encode()).hexdigest()[0] >= 'c' else 'train'
    gts = (s.get('detections') or {}).get('detections') or []
    want = [g for g in gts if (g['label'] in ('car', 'truck', 'bus') and not LIGHTS_ONLY) or (g['label'] == 'traffic light' and g.get('trafficLightColor') in ('red', 'R'))]
    if not want: continue
    im = None; got = 0
    tod = (s.get('timeofday') or {}).get('label')
    for gi, g in enumerate(want):
        if got >= mpi: break
        if im is None: im = Image.open(path).convert('RGB'); A = np.asarray(im); H, W = A.shape[:2]
        x, y, w, h = g['bounding_box']; x0, y0, x1, y1 = int(x * W), int(y * H), int((x + w) * W) + 1, int((y + h) * H) + 1
        if x1 - x0 < 12 or y1 - y0 < 12: continue
        for bx0, by0, bx1, by1, npx in blobs(A[y0:y1, x0:x1]):
            bw, bh = bx1 - bx0, by1 - by0
            if bw < 6 or bh < 4 or bw > 0.6 * (x1 - x0) and g['label'] != 'traffic light': continue
            # detector-like box: blob plus its glow (x1.4), squarish
            cx, cy = x0 + (bx0 + bx1) / 2, y0 + (by0 + by1) / 2; s_ = 1.4 * max(bw, bh * 1.0)
            bw2, bh2 = max(s_, 1.4 * bw), max(s_ * 0.6, 1.4 * bh)
            b = [cx - bw2 / 2, cy - bh2 / 2, cx + bw2 / 2, cy + bh2 / 2]
            m = 0.15; ww, hh = b[2] - b[0], b[3] - b[1]
            cp = f"crops/{iid}_{gi}_{bx0}_{by0}.png"
            im.crop((int(b[0] - m * ww), int(b[1] - m * hh), int(b[2] + m * ww + 1), int(b[3] + m * hh + 1))).save(out + '/' + cp)
            rows.append(dict(img=iid, split=split, src=g['label'], tod=tod, box=[round(v, 1) for v in b], crop=cp))
            got += 1
            if got >= mpi: break
    if si % 1000 == 0: print(si, len(rows), flush=True)
json.dump(rows, open(out + '/blobs.json', 'w'))
print('done', len(rows))
