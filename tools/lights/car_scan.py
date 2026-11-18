#!/usr/bin/env python3
"""car_scan.py OUT.npz SESSION_DIR... : run the light detector (ONNX, the two views of the signd unit) and
the state classifier on the recorded frames of drive-recorder sessions (roadsense recorder: frames/<seq>.jpg,
1280x720, one a second) and store every detection with its native crop, the state CNN's probabilities and
the colour rule's scores. Input for the state-classifier evaluation on the car's own camera
(tools/lights/car_state_eval.py). Mirrors signd/lights.cc (letterbox top-left, grey 114, NMS 0.45)."""
import glob, json, os, sys
import numpy as np, cv2, onnxruntime as ort

DET = os.environ.get("LIGHTS_DET", "models/bdd100k_yolo11n_320x960.onnx")
STATE = os.environ.get("LIGHTS_STATE", "models/lights_state_bdd_48x24.onnx")
ROIS = [(0.0, 0.0, 1.0, 0.62), (0.125, 0.125, 0.875, 0.569)]
IN_W, IN_H, THR, NMS, CLS = 960, 320, float(os.environ.get("LIGHTS_THR", 0.12)), 0.45, 8
RULE = dict(v_min=0.45, v_rel=0.6, s_min=0.35, rule_min=0.004)

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)

def detect(sess, rgb, roi):
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = int(np.floor(roi[0] * W)), int(np.floor(roi[1] * H)), int(np.ceil(roi[2] * W)), int(np.ceil(roi[3] * H))
    r = min(IN_W / (x1 - x0), IN_H / (y1 - y0))
    nw, nh = min(IN_W, round((x1 - x0) * r)), min(IN_H, round((y1 - y0) * r))
    inp = np.full((IN_H, IN_W, 3), 114, np.uint8)
    inp[:nh, :nw] = cv2.resize(rgb[y0:y1, x0:x1], (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    o = sess.run(None, {sess.get_inputs()[0].name: (inp.astype(np.float32) / 255).transpose(2, 0, 1)[None]})[0][0]
    sc = o[4 + CLS]; keep = np.where(sc > THR)[0]
    out = []
    for i in keep:
        cx, cy, w, h = o[0, i], o[1, i], o[2, i], o[3, i]
        b = [max(0, x0 + (cx - w / 2) / r), max(0, y0 + (cy - h / 2) / r), min(W, x0 + (cx + w / 2) / r), min(H, y0 + (cy + h / 2) / r)]
        if b[3] - b[1] >= 6 and b[2] - b[0] >= 2: out.append((float(sc[i]), b))
    return out

def nms(c):
    c.sort(key=lambda x: -x[0]); out = []
    for s, b, v in c:
        if all(iou(b, o[1]) <= NMS for o in out): out.append((s, b, v))
    return out

def rule_scores(rgb, horizontal):
    x = rgb.astype(np.float32) / 255; h, w = x.shape[:2]
    mx = x.max(2); mn = x.min(2); d = mx - mn + 1e-6
    thr = max(RULE["v_min"], RULE["v_rel"] * np.sort(mx.ravel())[int(0.99 * (mx.size - 1))])
    s = (mx - mn) / (mx + 1e-6)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    hue = np.where(mx == r, ((g - b) / d + 6) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    lit = (mx >= thr) & (s >= RULE["s_min"])
    pos = np.broadcast_to(((np.arange(w)[None, :] + 0.5) / w) if horizontal else ((np.arange(h)[:, None] + 0.5) / h), (h, w))
    out = []
    for c, m in enumerate([(hue >= 345) | (hue < 20), (hue >= 20) & (hue < 60), (hue >= 130) & (hue < 210)]):
        pw = np.clip(1 - np.abs(pos - (1 / 6, 0.5, 5 / 6)[c]) * 1.5, 0, 1)
        out.append(float((s * mx * pw * (lit & m)).sum() / (h * w)))
    return out

def state_input(rgb, box):
    x0, y0, x1, y1 = box
    crop = rgb[int(round(y0)):max(int(round(y0)) + 3, int(round(y1))), int(round(x0)):max(int(round(x0)) + 2, int(round(x1)))]
    horizontal = crop.shape[1] > 1.2 * crop.shape[0]
    if horizontal: crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop, horizontal

def main():
    out, sessions = sys.argv[1], sys.argv[2:]
    so = ort.SessionOptions(); so.intra_op_num_threads = int(os.environ.get("THREADS", 4))
    det = ort.InferenceSession(DET, so, providers=["CPUExecutionProvider"]); st = ort.InferenceSession(STATE, so, providers=["CPUExecutionProvider"])
    meta, crops = [], []
    every = int(os.environ.get("EVERY", 1))
    for s in sessions:
        files = sorted(glob.glob(os.path.join(s, "frames", "*.jpg")), key=lambda p: int(os.path.basename(p)[:-4]))[::every]
        for n, f in enumerate(files):
            bgr = cv2.imread(f)
            if bgr is None: continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            c = []
            for v, roi in enumerate(ROIS): c += [(sc, b, v) for sc, b in detect(det, rgb, roi)]
            for sc, b, v in nms(c):
                crop, hor = state_input(rgb, b)
                x = cv2.resize(crop, (24, 48), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
                lg = st.run(None, {st.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0]
                p = np.exp(lg - lg.max()); p /= p.sum()
                bw, bh = b[2] - b[0], b[3] - b[1]; m = 0.5
                X0, Y0 = int(max(0, b[0] - m * bw)), int(max(0, b[1] - m * bh)); X1, Y1 = int(min(rgb.shape[1], b[2] + m * bw)), int(min(rgb.shape[0], b[3] + m * bh))
                meta.append(dict(session=os.path.basename(s.rstrip("/")), seq=int(os.path.basename(f)[:-4]), box=[round(float(q), 1) for q in b], score=round(sc, 3), view=v,
                                 horizontal=bool(hor), cnn=[round(float(q), 4) for q in p], rule=[round(q, 5) for q in rule_scores(crop if not hor else cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE), hor)],
                                 ctx=[X0, Y0, X1, Y1], mean_v=round(float(rgb[int(b[1]):int(b[3]) + 1, int(b[0]):int(b[2]) + 1].max(2).mean()) / 255, 3)))
                crops.append(rgb[Y0:Y1, X0:X1].copy())
            if n % 200 == 0: print(os.path.basename(s), n, len(files), len(meta), flush=True)
    np.savez_compressed(out, crops=np.array(crops, dtype=object), meta=json.dumps(meta))
    print("detections", len(meta))
main()
