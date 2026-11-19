#!/usr/bin/env python3
"""car_scan.py OUT.npz SESSION_DIR... : the road-user classes of the light detector on recorded frames.

Runs the BDD100K YOLO11n of `signd --lights` (ONNX, 320x960) on the drive recorder's frames (roadsense recorder:
frames/<seq>.jpg, 1280x720, one a second) in the views of the phone's unit, and stores every box of the eight
road-user classes (person, rider, car, truck, bus, train, motor, bike) above a low score with all eight class
scores, so that thresholds and the pairing of riders can be chosen afterwards (tools/road_users/score.py).
Mirrors signd/lights.cc: region of interest letterboxed top-left into the input, grey 114, class-agnostic NMS.

Views (VIEWS=0,1 by default; 2 is the candidate extra pass evaluated in roadsense docs/lanes/L42-road-users.md):
  0  the unit's wide view  (0, 0, 1, 0.62)            every other processed frame
  1  the unit's second view (0.125, 0.125, 0.875, 0.569)  every other processed frame
  2  a lower band          (0, 0.33, 1, 0.95)          not run by the unit (a dedicated pass would cost one more
                                                     detector run)
Environment: LIGHTS_DET (model path), THREADS, EVERY (every n-th frame), THR (keep threshold, default 0.15)."""
import glob, os, sys
import numpy as np, cv2, onnxruntime as ort

DET = os.environ.get("LIGHTS_DET", "models/bdd100k_yolo11n_320x960.onnx")
ROIS = [(0.0, 0.0, 1.0, 0.62), (0.125, 0.125, 0.875, 0.569), (0.0, 0.33, 1.0, 0.95)]
IN_W, IN_H, NMS = 960, 320, 0.45
THR = float(os.environ.get("THR", 0.15))
RU = list(range(8))          # BDD100K ids 0..7: person rider car truck bus train motor bike


def detect(sess, rgb, roi):
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = int(np.floor(roi[0] * W)), int(np.floor(roi[1] * H)), int(np.ceil(roi[2] * W)), int(np.ceil(roi[3] * H))
    r = min(IN_W / (x1 - x0), IN_H / (y1 - y0))
    nw, nh = min(IN_W, round((x1 - x0) * r)), min(IN_H, round((y1 - y0) * r))
    inp = np.full((IN_H, IN_W, 3), 114, np.uint8)
    inp[:nh, :nw] = cv2.resize(rgb[y0:y1, x0:x1], (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    o = sess.run(None, {sess.get_inputs()[0].name: (inp.astype(np.float32) / 255).transpose(2, 0, 1)[None]})[0][0]
    sc = o[4:4 + 8]                         # [8, N]
    best = sc.max(0)
    keep = np.where(best > THR)[0]
    cand = []
    for i in keep:
        cx, cy, w, h = o[0, i], o[1, i], o[2, i], o[3, i]
        b = [max(0.0, x0 + (cx - w / 2) / r), max(0.0, y0 + (cy - h / 2) / r),
             min(float(W), x0 + (cx + w / 2) / r), min(float(H), y0 + (cy + h / 2) / r)]
        if b[2] - b[0] < 3 or b[3] - b[1] < 3:
            continue
        cand.append((float(best[i]), b, sc[:, i].astype(np.float32)))
    cand.sort(key=lambda c: -c[0])
    out = []
    for s, b, v in cand:
        if all(iou(b, o[1]) <= NMS for o in out):
            out.append((s, b, v))
    return out, (x0, y0, x1, y1)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def main():
    out, sessions = sys.argv[1], sys.argv[2:]
    views = [int(v) for v in os.environ.get("VIEWS", "0,1,2").split(",")]
    so = ort.SessionOptions(); so.intra_op_num_threads = int(os.environ.get("THREADS", 4))
    det = ort.InferenceSession(DET, so, providers=["CPUExecutionProvider"])
    rows, scores, frames = [], [], []
    every = int(os.environ.get("EVERY", 1))
    for s in sessions:
        files = sorted(glob.glob(os.path.join(s, "frames", "*.jpg")), key=lambda p: int(os.path.basename(p)[:-4]))[::every]
        for n, f in enumerate(files):
            bgr = cv2.imread(f)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            fi = len(frames)
            frames.append(f)
            for v in views:
                dets, roi = detect(det, rgb, ROIS[v])
                for sc, b, vec in dets:
                    rows.append([fi, v, sc] + b + list(roi))
                    scores.append(vec)
            if n % 200 == 0:
                print(s, n, len(files), len(rows), file=sys.stderr, flush=True)
    np.savez_compressed(out, rows=np.array(rows, np.float32).reshape(-1, 11), scores=np.array(scores, np.float32).reshape(-1, 8),
                        frames=np.array(frames), views=np.array(views), rois=np.array(ROIS, np.float32))
    print("frames", len(frames), "boxes", len(rows), file=sys.stderr)


if __name__ == "__main__":
    main()
