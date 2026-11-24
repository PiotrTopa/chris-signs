#!/usr/bin/env python3
"""crops_extract.py MODEL.onnx RIDE.mjpeg OUT.npz FIRST LAST [EVERY]: full-frame detector (bdd100k_yolo11n_384x640.onnx:
tools/export_onnx.py ... 384 640) on every EVERY-th
frame; per kept road-user box: frame, class, score, box and a 32x32 RGB crop (box + 5 % margin)."""
import sys, numpy as np, cv2, onnxruntime as ort
model, src, out, A, B = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
every = int(sys.argv[6]) if len(sys.argv) > 6 else 2
so = ort.SessionOptions(); so.intra_op_num_threads = 3
sess = ort.InferenceSession(model, so, providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0].name
THR = np.array([0.40, 0.35, 0.40, 0.40, 0.40, 0.50, 0.35, 0.35])
rows, crops = [], []
buf, i = b"", 0
f = open(src, 'rb'); done = False
while not done:
    chunk = f.read(1 << 22)
    if not chunk: break
    buf += chunk
    while True:
        s = buf.find(b"\xff\xd8")
        if s < 0: buf = b""; break
        e = buf.find(b"\xff\xd9", s + 2)
        if e < 0: buf = buf[s:]; break
        jpg = buf[s:e + 2]; buf = buf[e + 2:]
        if i > B: done = True; break
        if i >= A and (i - A) % every == 0:
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            H, W = img.shape[:2]
            r = min(640 / W, 384 / H); nw, nh = round(W * r), round(H * r)
            x = np.full((384, 640, 3), 114, np.uint8); x[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            t = (x[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)) / 255
            y = sess.run(None, {inp: t})[0][0]
            p = y[4:12]; c = p.argmax(0); sc = p.max(0)
            keep = np.where(sc > THR[c])[0]; keep = keep[np.argsort(-sc[keep])]
            bx = np.stack([y[0] - y[2] / 2, y[1] - y[3] / 2, y[0] + y[2] / 2, y[1] + y[3] / 2], 1) / r
            sel = []
            for k in keep:
                ok = True
                for j in sel:
                    ix = max(0, min(bx[k, 2], bx[j, 2]) - max(bx[k, 0], bx[j, 0])); iy = max(0, min(bx[k, 3], bx[j, 3]) - max(bx[k, 1], bx[j, 1]))
                    u = ix * iy; u = u / ((bx[k, 2] - bx[k, 0]) * (bx[k, 3] - bx[k, 1]) + (bx[j, 2] - bx[j, 0]) * (bx[j, 3] - bx[j, 1]) - u + 1e-9)
                    if u > 0.5: ok = False; break
                if ok: sel.append(k)
            for k in sel:
                b = bx[k]; w, h = b[2] - b[0], b[3] - b[1]
                if w < 8 or h < 8: continue
                x0, y0, x1, y1 = int(max(0, b[0] - .05 * w)), int(max(0, b[1] - .05 * h)), int(min(W, b[2] + .05 * w + 1)), int(min(H, b[3] + .05 * h + 1))
                # overlap with other kept boxes (for clean training pairs)
                ov = 0.0
                for j in sel:
                    if j == k: continue
                    ix = max(0, min(b[2], bx[j, 2]) - max(b[0], bx[j, 0])); iy = max(0, min(b[3], bx[j, 3]) - max(b[1], bx[j, 1]))
                    ov = max(ov, ix * iy / (w * h))
                rows.append([i, c[k], sc[k], *b, ov])
                crops.append(cv2.resize(img[y0:y1, x0:x1], (32, 32), interpolation=cv2.INTER_AREA))
            if ((i - A) // every) % 500 == 0: print(i, len(rows), flush=True)
        i += 1
np.savez_compressed(out, rows=np.array(rows, np.float32), crops=np.array(crops, np.uint8))
print('done', len(rows))
