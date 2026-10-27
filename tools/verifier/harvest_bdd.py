#!/usr/bin/env python3
"""Run the signd mirror on BDD100K val (FiftyOne export); keep detections whose classifier top-1 is a round red class,
with their overlap against BDD GT boxes, and save crops. usage: harvest_bdd.py BDD_DIR OUTDIR [threads]"""
import sys, os, json, hashlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipe
from PIL import Image
bdd, out = sys.argv[1], sys.argv[2]; thr = int(sys.argv[3]) if len(sys.argv) > 3 else 4
os.makedirs(out + '/crops', exist_ok=True)
S = json.load(open(bdd + '/samples.json'))['samples']
P = pipe.Pipe(thr)
def ov(a, b):  # intersection / area(a), IoU
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    aa = (a[2] - a[0]) * (a[3] - a[1]); bb = (b[2] - b[0]) * (b[3] - b[1])
    return i / max(aa, 1e-6), i / max(aa + bb - i, 1e-6)
with open(out + '/dets.jsonl', 'w') as f:
    for n, s in enumerate(S):
        path = bdd + '/' + s['filepath']; im = Image.open(path).convert('RGB'); W, H = im.size
        gts = []
        for g in (s.get('detections') or {}).get('detections') or []:
            x, y, w, h = g['bounding_box']; gts.append((g['label'], [x * W, y * H, (x + w) * W, (y + h) * H], g.get('trafficLightColor')))
        d = P.detect(im); probs = P.classify(im, [b for b, _ in d])
        iid = os.path.basename(path)[:-4]
        split = 'test' if hashlib.md5(iid.encode()).hexdigest()[0] >= 'c' else 'train'
        for k, ((b, sc), p) in enumerate(zip(d, probs)):
            top = np.argsort(-p)[:5]
            if not pipe.is_red(pipe.NAMES[top[0]]): continue
            best = {}
            for lab, gb, col in gts:
                inA, iou = ov(b, gb)
                if inA > best.get(lab, (0,))[0]: best[lab] = (round(inA, 3), round(iou, 3), [round(v) for v in gb], col)
            w, h = b[2] - b[0], b[3] - b[1]; m = 0.15
            cp = f"crops/{iid}_{k}.png"
            im.crop((int(b[0] - m * w), int(b[1] - m * h), int(b[2] + m * w + 1), int(b[3] + m * h + 1))).save(out + '/' + cp)
            f.write(json.dumps(dict(img=iid, split=split, k=k, W=W, H=H, box=[round(v, 1) for v in b], score=round(sc, 4),
                                    top=[[int(t), round(float(p[t]), 4)] for t in top], cls=pipe.MERGED[top[0]], gt=best,
                                    tod=(s.get('timeofday') or {}).get('label') if isinstance(s.get('timeofday'), dict) else s.get('timeofday'),
                                    crop=cp)) + '\n')
        if n % 500 == 0: print(n, flush=True)
