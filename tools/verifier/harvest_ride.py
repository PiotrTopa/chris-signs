#!/usr/bin/env python3
"""Run the signd mirror on every Nth frame of the Krakow mjpeg; write dets.jsonl and crops of red-class detections.
usage: harvest_ride.py MJPEG OUTDIR [every] [threads]"""
import sys, os, io, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipe
from PIL import Image
src, out = sys.argv[1], sys.argv[2]; every = int(sys.argv[3]) if len(sys.argv) > 3 else 1
thr = int(sys.argv[4]) if len(sys.argv) > 4 else 4
os.makedirs(out + '/crops', exist_ok=True)
mm, st = pipe.mjpeg_index(src); st.append(len(mm))
P = pipe.Pipe(thr)
with open(out + '/dets.jsonl', 'w') as f:
    for i in range(0, len(st) - 1, every):
        im = Image.open(io.BytesIO(mm[st[i]:st[i + 1]])).convert('RGB')
        d = P.detect(im); probs = P.classify(im, [b for b, s in d])
        for k, ((b, s), p) in enumerate(zip(d, probs)):
            top = np.argsort(-p)[:5]
            rec = dict(f=i, k=k, box=[round(v, 1) for v in b], score=round(s, 4), top=[[int(t), round(float(p[t]), 4)] for t in top],
                       cls=pipe.MERGED[top[0]], red=pipe.is_red(pipe.NAMES[top[0]]))
            if rec['red']:
                w, h = b[2] - b[0], b[3] - b[1]; m = 0.15
                im.crop((int(b[0] - m * w), int(b[1] - m * h), int(b[2] + m * w + 1), int(b[3] + m * h + 1))).save(f"{out}/crops/{i:05d}_{k}.png")
            f.write(json.dumps(rec) + '\n')
        if i % 1000 == 0: print(i, flush=True)
