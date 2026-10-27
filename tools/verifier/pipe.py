"""Python mirror of signd's detector + classifier (ORT, PIL BILINEAR, as tools/ncnn/compare_models.py)."""
import numpy as np, onnxruntime as ort
from PIL import Image
import os as _os
M = _os.environ.get('SIGND_MODELS', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'models')) + '/'
NAMES = [l.strip() for l in open(M + 'mtsd_signcls_classes.txt') if l.strip()]
def strip(n):
    import re
    return re.sub(r'--g\d+$', '', n)
MERGED = [strip(n) for n in NAMES]
import re
RED_RE = re.compile(r'^regulatory--(no-|maximum-speed-limit|road-closed|height-limit|weight-limit|width-limit|stop$|stop--)')
def is_red(name): return bool(RED_RE.match(strip(name)))

class Pipe:
    def __init__(self, threads=4):
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
        self.det = ort.InferenceSession(M + 'mtsd_signdet_yolov8n_384x640.onnx', so, providers=['CPUExecutionProvider'])
        self.cls = ort.InferenceSession(M + 'mtsd_signcls_yolov8n_64.onnx', so, providers=['CPUExecutionProvider'])
    def letterbox(self, im, W=640, H=384):
        r = min(W / im.width, H / im.height)
        nw, nh = round(im.width * r), round(im.height * r)
        left, top = round((W - nw) / 2 - 0.1), round((H - nh) / 2 - 0.1)
        can = Image.new('RGB', (W, H), (114, 114, 114)); can.paste(im.resize((nw, nh), Image.BILINEAR), (left, top))
        return np.asarray(can, dtype=np.float32).transpose(2, 0, 1)[None] / 255, r, left, top
    def detect(self, im, thr=0.35, iou=0.45):
        x, r, left, top = self.letterbox(im)
        o = self.det.run(None, {self.det.get_inputs()[0].name: x})[0].reshape(5, -1)
        keep = o[4] > thr; b = o[:4, keep].T; s = o[4, keep]
        boxes = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        order = np.argsort(-s); out = []
        while len(order):
            i = order[0]; out.append((boxes[i], float(s[i]))); rest = order[1:]
            ix = np.maximum(0, np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]))
            iy = np.maximum(0, np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]))
            inter = ix * iy
            ua = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) + (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1]) - inter
            order = rest[inter / (ua + 1e-9) <= iou]
        res = []
        for bb, sc in out[:100]:
            fx0, fy0, fx1, fy1 = (bb[0] - left) / r, (bb[1] - top) / r, (bb[2] - left) / r, (bb[3] - top) / r
            fx0, fy0 = max(0., fx0), max(0., fy0); fx1, fy1 = min(im.width, fx1), min(im.height, fy1)
            if fx1 - fx0 < 4 or fy1 - fy0 < 4: continue
            res.append(([float(fx0), float(fy0), float(fx1), float(fy1)], sc))
        return res
    def cls_input(self, im, box):
        fx0, fy0, fx1, fy1 = box; w, h = fx1 - fx0, fy1 - fy0
        s = 64 / min(w, h); rw, rh = max(64, round(w * s)), max(64, round(h * s))
        crop = im.crop((fx0, fy0, fx1, fy1)).resize((rw, rh), Image.BILINEAR)
        l, t = (rw - 64) // 2, (rh - 64) // 2
        return np.asarray(crop.crop((l, t, l + 64, t + 64)), dtype=np.float32).transpose(2, 0, 1) / 255
    def classify(self, im, boxes):
        if not boxes: return np.zeros((0, len(NAMES)), np.float32)
        x = np.stack([self.cls_input(im, b) for b in boxes])
        outs = [self.cls.run(None, {self.cls.get_inputs()[0].name: x[i:i+1]})[0] for i in range(len(x))]
        return np.concatenate(outs).reshape(len(boxes), -1)

def mjpeg_index(path):
    import mmap
    f = open(path, 'rb'); mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    starts = []; p = 0
    while True:
        p = mm.find(b'\xff\xd8\xff', p)
        if p < 0: break
        starts.append(p); p += 3
    return mm, starts
