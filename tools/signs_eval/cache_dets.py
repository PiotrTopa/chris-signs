#!/usr/bin/env python3
"""Per-frame sign detections of a ride for the offline evaluation of lane L24-S (laptop GPU).

For every requested frame of a ride (MJPEG + JPEG index) and every requested view, runs the sign detector
exactly as signd's SignPipeline::process does (same NV12 conversion and resampler through libsigns_pre.so,
same letterbox, decode, NMS, box mapping, classifier crop, round-red appearance and verifier), with the ONNX
graphs evaluated by tools/signs_eval/onnx_torch.py on the GPU. Writes one file per view:

  OUT/<slug>.<view>.npz   frames (processed frame indices), and per detection: frame, box[4], score, cls_raw,
                          cls_prob, glow, verify, probs (float32, only for score > --probs-min, else empty row)
  OUT/<slug>.<view>.dets  the same in the binary layout tools/signs_track_replay.cc reads

Views: A = the whole frame (signd's letterbox); B=x0,y0,x1,y1 = a window (frame fractions, signd --roi2);
T = the reference tiles (640x384 windows at scale 1.0: x0 0/320/640, y0 0/336) merged with A by NMS.
usage: cache_dets.py SLUG OUT --frames START:STOP:STEP --view A --view B=0.5,0.2,1,0.733 [--det-thr 0.2]
"""
import argparse
import ctypes
import glob
import os
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_torch import OnnxTorch  # noqa: E402

H = os.path.expanduser("~")
W_, H_ = 1280, 720
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
POOL = ThreadPoolExecutor(4)


def load_lib(path):
    lib = ctypes.CDLL(path)
    lib.pre_jpeg_nv12.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_int)]
    lib.pre_region.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int] + [ctypes.c_double] * 4 + \
        [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.pre_appearance.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int] + [ctypes.c_float] * 4 + [ctypes.c_void_p]
    lib.pre_appearance.restype = ctypes.c_float
    lib.pre_is_round_red.argtypes = [ctypes.c_char_p]
    return lib


def strip_variant(n):
    p = n.rfind("--g")
    if p >= 0 and p + 3 < len(n) and n[p + 3:].isdigit():
        return n[:p]
    return n


class Frames:
    def __init__(self, slug):
        self.src = glob.glob(f"{H}/datasets/rides/{slug}/*.mjpeg")[0]
        ip = f"{H}/l11/frames/{slug}/full/index.npy"
        if not os.path.exists(ip):
            ip = f"{H}/datasets/l24s/index/{slug}.npy"
            if not os.path.exists(ip):
                os.makedirs(os.path.dirname(ip), exist_ok=True)
                np.save(ip, build_index(self.src))
        self.idx = np.load(ip)
        self.f = open(self.src, "rb")

    def __len__(self):
        return len(self.idx) - 1

    def jpeg(self, k):
        self.f.seek(self.idx[k])
        return self.f.read(self.idx[k + 1] - self.idx[k])


def build_index(src):
    offs, pos, chunk = [], 0, 64 << 20
    with open(src, "rb") as f:
        tail = b""
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            data = tail + buf
            base = pos - len(tail)
            i = 0
            while True:
                j = data.find(b"\xff\xd8\xff", i)
                if j < 0 or j >= len(data) - 3:
                    break
                offs.append(base + j)
                i = j + 3
            tail = data[-3:]
            pos += len(buf)
    return np.array(offs + [os.path.getsize(src)], dtype=np.int64)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-9)


def nms_greedy(boxes, scores, thr, max_det):
    """Greedy NMS as signd (descending score, stable; IoU > thr suppresses). -> kept indices into the input."""
    order = np.argsort(-scores, kind="stable")
    b = boxes[order].astype(np.float32)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    dead = np.zeros(len(order), bool)
    kept = []
    for i in range(len(order)):
        if dead[i]:
            continue
        kept.append(order[i])
        if len(kept) >= max_det:
            break
        ix = np.maximum(0, np.minimum(b[i, 2], b[i + 1:, 2]) - np.maximum(b[i, 0], b[i + 1:, 0]))
        iy = np.maximum(0, np.minimum(b[i, 3], b[i + 1:, 3]) - np.maximum(b[i, 1], b[i + 1:, 1]))
        inter = ix * iy
        u = area[i] + area[i + 1:] - inter
        dead[i + 1:] |= inter / (u + np.float32(1e-9)) > thr
    return kept


class Detector:
    """SignPipeline's detector stage (letterbox of a frame rectangle, decode, NMS, mapping), batched on the GPU."""

    def __init__(self, lib, det_path, in_w=640, in_h=384, det_thr=0.2, nms=0.45, max_det=100, min_box=4.0,
                 score_row=4, top_left=False):
        self.lib, self.in_w, self.in_h = lib, in_w, in_h
        self.det = OnnxTorch(det_path)
        self.det_thr, self.nms, self.max_det, self.min_box = det_thr, nms, max_det, min_box
        self.score_row, self.top_left = score_row, top_left  # light detector: class row 4 + 9, content top-left

    def geom(self, rect):
        rx0, ry0, rx1, ry1 = rect
        rw, rh = rx1 - rx0, ry1 - ry0
        r = min(self.in_w / rw, self.in_h / rh)
        if self.top_left:  # LightPipeline::process
            return r, min(self.in_w, int(round_half_away(rw * r))), min(self.in_h, int(round_half_away(rh * r))), 0, 0
        nw, nh = int(round_half_away(rw * r)), int(round_half_away(rh * r))
        left = int(round_half_away((self.in_w - nw) / 2.0 - 0.1))
        top = int(round_half_away((self.in_h - nh) / 2.0 - 0.1))
        return r, nw, nh, left, top

    def run_many(self, jobs):
        """jobs: [(nv12, rect, drop_inner_edges)] -> [[(score, [x0, y0, x1, y1])]] per job."""
        if not jobs:
            return []
        x = np.full((len(jobs), 3, self.in_h, self.in_w), np.float32(114.0 / 255.0), np.float32)
        geo = [None] * len(jobs)

        def prep(j):
            nv12, rect, _ = jobs[j]
            r, nw, nh, left, top = self.geom(rect)
            geo[j] = (r, left, top)
            self.lib.pre_region(nv12.ctypes.data, W_, H_, rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1], nw, nh,
                                x[j].ctypes.data, self.in_w, self.in_h, left, top)
        list(POOL.map(prep, range(len(jobs))))
        res = []
        outs = []
        for k in range(0, len(jobs), 16):
            outs.append(self.det(torch.from_numpy(x[k:k + 16]).cuda())[0].float().cpu().numpy())
        out = np.concatenate(outs)
        for j, (nv12, rect, inner) in enumerate(jobs):
            o = out[j]
            r, left, top = geo[j]
            rx0, ry0, rx1, ry1 = rect
            keep = np.nonzero(o[self.score_row] > np.float32(self.det_thr))[0]
            if len(keep) == 0:
                res.append([])
                continue
            cx, cy, w, h = o[0, keep], o[1, keep], o[2, keep], o[3, keep]
            boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).astype(np.float32)
            sc = o[self.score_row, keep]
            dets = []
            for i in nms_greedy(boxes, sc, self.nms, self.max_det):
                b = boxes[i].astype(np.float64)
                x0 = np.float32(min(max(rx0 + (b[0] - left) / r, rx0), min(W_, rx1)))
                x1 = np.float32(min(max(rx0 + (b[2] - left) / r, rx0), min(W_, rx1)))
                y0 = np.float32(min(max(ry0 + (b[1] - top) / r, ry0), min(H_, ry1)))
                y1 = np.float32(min(max(ry0 + (b[3] - top) / r, ry0), min(H_, ry1)))
                if x1 - x0 < self.min_box or y1 - y0 < self.min_box:
                    continue
                if inner:
                    m = 2.0
                    if (rx0 > 0 and x0 <= rx0 + m) or (rx1 < W_ and x1 >= rx1 - m) or (ry0 > 0 and y0 <= ry0 + m) or \
                            (ry1 < H_ and y1 >= ry1 - m):
                        continue
                dets.append((float(sc[i]), [float(x0), float(y0), float(x1), float(y1)]))
            res.append(dets)
        return res


def round_half_away(x):
    return np.floor(x + 0.5) if x >= 0 else -np.floor(-x + 0.5)


class Classifier:
    def __init__(self, lib, cls_path, names_path, ver_path, cls_s=64):
        self.lib, self.s = lib, cls_s
        self.cls = OnnxTorch(cls_path)
        self.ver = OnnxTorch(ver_path) if ver_path else None
        self.raw = [l.rstrip("\r ").rstrip("\n").rstrip("\r ") for l in open(names_path)]
        self.raw = [l for l in self.raw]
        self.red = [bool(lib.pre_is_round_red(strip_variant(n).encode())) for n in self.raw]

    def run(self, nv12, boxes):
        n = len(boxes)
        if n == 0:
            return np.zeros((0, len(self.raw)), np.float32), [], []
        s = self.s
        x = np.empty((n, 3, s, s), np.float32)
        for i, b in enumerate(boxes):
            x0, y0, x1, y1 = max(0.0, b[0]), max(0.0, b[1]), min(float(W_), b[2]), min(float(H_), b[3])
            w, h = x1 - x0, y1 - y0
            sc = s / min(w, h)
            rw = max(s, int(round_half_away(w * sc)))
            rh = max(s, int(round_half_away(h * sc)))
            l, tp = (rw - s) // 2, (rh - s) // 2
            sxs, sys_ = w / rw, h / rh
            self.lib.pre_region(nv12.ctypes.data, W_, H_, x0 + l * sxs, y0 + tp * sys_, s * sxs, s * sys_, s, s,
                                x[i].ctypes.data, s, s, 0, 0)
        p = self.cls(torch.from_numpy(x).cuda())[0].float().cpu().numpy()
        glow, ver = [], []
        app = np.empty((1, 3, 32, 32), np.float32)
        for i, b in enumerate(boxes):
            k = int(np.argmax(p[i]))
            if self.red[k]:
                glow.append(float(self.lib.pre_appearance(nv12.ctypes.data, W_, H_, b[0], b[1], b[2], b[3],
                                                          app.ctypes.data)))
                if self.ver is not None:
                    lg = float(self.ver(app)[0].float().cpu().numpy().reshape(-1)[0])
                    ver.append(1.0 / (1.0 + np.exp(-lg)))
                else:
                    ver.append(-1.0)
            else:
                glow.append(-1.0)
                ver.append(-1.0)
        return p, glow, ver


def view_rects(spec):
    if spec == "A":
        return [((0.0, 0.0, float(W_), float(H_)), False)]
    if spec.startswith("B") and "=" in spec:
        f = [float(v) for v in spec.split("=")[1].split(",")]
        r = (max(0.0, np.floor(f[0] * W_)), max(0.0, np.floor(f[1] * H_)), min(float(W_), np.ceil(f[2] * W_)),
             min(float(H_), np.ceil(f[3] * H_)))
        return [(tuple(float(v) for v in r), True)]
    if spec.startswith("L") and "=" in spec:  # light detector window (frame fractions, floor/ceil)
        f = [float(v) for v in spec.split("=")[1].split(",")]
        r = (np.floor(f[0] * W_), np.floor(f[1] * H_), np.ceil(f[2] * W_), np.ceil(f[3] * H_))
        return [(tuple(float(v) for v in r), False)]
    if spec == "T3":  # one row of full-scale tiles around the horizon band
        return [((float(x0), 168.0, float(x0 + 640), 552.0), True) for x0 in (0, 320, 640)]
    if spec in ("T", "T6"):
        out = [] if spec == "T6" else [((0.0, 0.0, float(W_), float(H_)), False)]
        for y0 in (0, 336):
            for x0 in (0, 320, 640):
                out.append(((float(x0), float(y0), float(x0 + 640), float(y0 + 384)), True))
        return out
    raise SystemExit("bad view " + spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("out")
    ap.add_argument("--frames", required=True, help="START:STOP:STEP of ride frames (STOP may be empty)")
    ap.add_argument("--view", action="append", required=True)
    ap.add_argument("--name", action="append", help="file name per view (default A, B, T)")
    ap.add_argument("--det-thr", type=float, default=0.2)
    ap.add_argument("--probs-min", type=float, default=0.35)
    ap.add_argument("--lib", default=f"{H}/datasets/l24s/lib/libsigns_pre.so")
    ap.add_argument("--det", default=f"{H}/projects/roadsense-signs/models/mtsd_signdet_yolov8n_384x640.onnx")
    ap.add_argument("--cls", default=f"{H}/l12/cls1/signcls_64.onnx")
    ap.add_argument("--cls-names", default=f"{H}/l12/cls1/classes.txt")
    ap.add_argument("--verifier", default=f"{H}/l8/models/verifier_v1.onnx")
    ap.add_argument("--light-det", default=f"{H}/l7/models/bdd11n_320x960.onnx")
    ap.add_argument("--light-cls", type=int, default=9, help="BDD100K class of the light detector views (9 = traffic sign)")
    ap.add_argument("--every-view", action="append", default=[],
                    help="NAME=K: compute this view only on frames that are multiples of K")
    a = ap.parse_args()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    lib = load_lib(a.lib)
    fr = Frames(a.slug)
    st, sp, stp = (a.frames.split(":") + ["", ""])[:3]
    frames = list(range(int(st or 0), min(int(sp), len(fr)) if sp else len(fr), int(stp or 1)))
    if not frames:
        raise SystemExit(0)
    det = Detector(lib, a.det, det_thr=a.det_thr)
    ldet = None
    if any(v.startswith("L") and "=" in v for v in a.view):
        ldet = Detector(lib, a.light_det, in_w=960, in_h=320, det_thr=a.det_thr, max_det=30, min_box=4.0,
                        score_row=4 + a.light_cls, top_left=True)
    cls = Classifier(lib, a.cls, a.cls_names, a.verifier)
    nc = len(cls.raw)
    names = [v.split("=")[0] for v in a.view]
    os.makedirs(a.out, exist_ok=True)
    views = [(nm, view_rects(v)) for nm, v in zip(names, a.view)]
    vdet = {nm: (ldet if v.startswith("L") and "=" in v else det) for nm, v in zip(names, a.view)}
    vevery = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.every_view}
    vframes = {nm: [k for k in frames if k % vevery.get(nm, 1) == 0] for nm, _ in views}
    acc = {nm: {"frame": [], "box": [], "score": [], "cls_raw": [], "cls_prob": [], "glow": [], "verify": [],
                "probs": []} for nm, _ in views}
    t0 = time.time()
    B = 8  # frames per GPU batch
    for n0 in range(0, len(frames), B):
        jps = [(k, fr.jpeg(k)) for k in frames[n0:n0 + B]]

        def decode(kj):
            k, jp = kj
            nv12 = np.empty(W_ * H_ * 3 // 2, np.uint8)
            w, h = ctypes.c_int(W_), ctypes.c_int(H_)
            if lib.pre_jpeg_nv12(jp, len(jp), nv12.ctypes.data, ctypes.byref(w), ctypes.byref(h)) != 0:
                print(f"frame {k}: decode failed ({w.value}x{h.value})", file=sys.stderr)
                return None
            return (k, nv12)
        chunk = [c for c in POOL.map(decode, jps) if c is not None]
        jobs, owner = {}, {}
        for k, nv12 in chunk:
            for nm, rects in views:
                if k % vevery.get(nm, 1):
                    continue
                for rect, inner in rects:
                    jobs.setdefault(id(vdet[nm]), []).append((nv12, rect, inner))
                    owner.setdefault(id(vdet[nm]), []).append((k, nm))
        per = {}
        for dk, jl in jobs.items():
            dd = det if dk == id(det) else ldet
            for (k, nm), d in zip(owner[dk], dd.run_many(jl)):
                per.setdefault((k, nm), []).extend(d)
        for k, nv12 in chunk:
            for nm, rects in views:
                if k % vevery.get(nm, 1):
                    continue
                dets = per.get((k, nm), [])
                if len(rects) > 1:  # merge views: NMS by score
                    dets.sort(key=lambda d: -d[0])
                    kept = []
                    for d in dets:
                        if all(iou(d[1], q[1]) <= 0.45 for q in kept):
                            kept.append(d)
                    dets = kept
                p, glow, ver = cls.run(nv12, [d[1] for d in dets])
                A = acc[nm]
                for i, d in enumerate(dets):
                    A["frame"].append(k)
                    A["box"].append(d[1])
                    A["score"].append(d[0])
                    kk = int(np.argmax(p[i]))
                    A["cls_raw"].append(kk)
                    A["cls_prob"].append(float(p[i][kk]))
                    A["glow"].append(glow[i])
                    A["verify"].append(ver[i])
                    A["probs"].append(p[i] if d[0] > a.probs_min else None)
        if (n0 // B) % 250 == 0:
            print(f"{a.slug}: {n0}/{len(frames)} frames, {time.time() - t0:.0f} s", file=sys.stderr)
    for nm, _ in views:
        A = acc[nm]
        has = np.array([q is not None for q in A["probs"]], bool)
        probs = np.array([q for q in A["probs"] if q is not None], np.float32).reshape(-1, nc)
        np.savez_compressed(f"{a.out}/{a.slug}.{nm}.npz", frames=np.array(vframes[nm], np.int32),
                            frame=np.array(A["frame"], np.int32), box=np.array(A["box"], np.float32).reshape(-1, 4),
                            score=np.array(A["score"], np.float32), cls_raw=np.array(A["cls_raw"], np.int32),
                            cls_prob=np.array(A["cls_prob"], np.float32), glow=np.array(A["glow"], np.float32),
                            verify=np.array(A["verify"], np.float32), has_probs=has)
        # binary for signs_track_replay: header, then per frame: frame, n, n x (8 floats, int, nc floats)
        with open(f"{a.out}/{a.slug}.{nm}.dets", "wb") as f:
            f.write(struct.pack("<4siii", b"L24D", 1, nc, len(vframes[nm])))
            pi = np.cumsum(has) - 1
            byf = {}
            for i, k in enumerate(A["frame"]):
                if has[i]:
                    byf.setdefault(k, []).append(i)
            for k in vframes[nm]:
                sel = byf.get(k, [])
                f.write(struct.pack("<ii", k, len(sel)))
                for i in sel:
                    b = A["box"][i]
                    f.write(struct.pack("<8fi", b[0], b[1], b[2], b[3], A["score"][i], A["cls_prob"][i], A["glow"][i],
                                        A["verify"][i], A["cls_raw"][i]))
                    f.write(probs[pi[i]].astype("<f4").tobytes())
    print(f"{a.slug}: {len(frames)} frames in {time.time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
