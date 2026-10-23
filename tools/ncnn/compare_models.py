#!/usr/bin/env python3
"""Laptop check of the ncnn conversion against the ONNX/ORT reference on real photos.
usage: compare_models.py DET.onnx DET-ncnn.param CLS.onnx CLS-ncnn.param IMG... [--vulkan] [--fp32]
Needs onnxruntime, ncnn (pip) and pillow (roadsense .venv-models has them). Preprocessing mirrors
signd (letterbox 384x640 with PIL BILINEAR, grey 114, /255; classifier: short side -> 64, centre
crop); detections decoded with score > 0.35, NMS 0.45. --vulkan runs ncnn on the laptop GPU
(fp16-packed unless --fp32), which is a first look at fp16 effects; the phone (turnip) is the real test.
The detector .param has two outputs (out0 boxes, out1 scores), see fix_ncnn.py."""
import sys, numpy as np, onnxruntime as ort, ncnn
from PIL import Image

args = [a for a in sys.argv[1:] if not a.startswith('--')]
det_onnx, det_param, cls_onnx, cls_param = args[:4]
imgs = args[4:]
vulkan = '--vulkan' in sys.argv
fp16 = '--fp32' not in sys.argv

def load_ncnn(param):
    net = ncnn.Net()
    net.opt.use_vulkan_compute = vulkan
    net.opt.use_fp16_packed = fp16; net.opt.use_fp16_storage = fp16; net.opt.use_fp16_arithmetic = fp16
    net.load_param(param); net.load_model(param[:-6] + '.bin')
    return net

def run_ncnn(net, x):  # x: [1,C,H,W] float32; concatenates out0, out1, ... like the ncnn runner
    ex = net.create_extractor()
    ex.input('in0', ncnn.Mat(np.ascontiguousarray(x[0])))
    parts = []
    for name in sorted(n for n in net.output_names() if n.startswith('out')):
        _, out = ex.extract(name)
        parts.append(np.array(out).reshape(-1, out.w) if out.dims >= 2 else np.array(out).reshape(1, -1))
    return np.vstack(parts) if parts[0].shape[1] == parts[-1].shape[1] else np.concatenate([p.ravel() for p in parts])

det_o = ort.InferenceSession(det_onnx, providers=['CPUExecutionProvider'])
cls_o = ort.InferenceSession(cls_onnx, providers=['CPUExecutionProvider'])
det_n, cls_n = load_ncnn(det_param), load_ncnn(cls_param)

def letterbox(im, W=640, H=384):
    r = min(W / im.width, H / im.height)
    nw, nh = round(im.width * r), round(im.height * r)
    left, top = round((W - nw) / 2 - 0.1), round((H - nh) / 2 - 0.1)
    can = Image.new('RGB', (W, H), (114, 114, 114))
    can.paste(im.resize((nw, nh), Image.BILINEAR), (left, top))
    return np.asarray(can, dtype=np.float32).transpose(2, 0, 1)[None] / 255, r, left, top

def decode(o, thr=0.35, iou=0.45):
    o = o.reshape(5, -1)
    keep = o[4] > thr
    b = o[:4, keep].T; s = o[4, keep]
    boxes = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
    order = np.argsort(-s); out = []
    while len(order):
        i = order[0]; out.append((boxes[i], s[i])); rest = order[1:]
        ix = np.maximum(0, np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]))
        iy = np.maximum(0, np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]))
        inter = ix * iy
        ua = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) + (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1]) - inter
        order = rest[inter / (ua + 1e-9) <= iou]
    return out


mode = ('vulkan ' if vulkan else 'cpu ') + ('fp16-packed' if fp16 else 'fp32')
print('ncnn mode:', mode)
agg = dict(det_raw_max=0, score_max=0, box_max=0, n_ref=0, n_match=0, cls_top1_same=0, cls_n=0, cls_prob_max=0)
for path in imgs:
    im = Image.open(path).convert('RGB')
    x, r, left, top = letterbox(im)
    o_ref = det_o.run(None, {det_o.get_inputs()[0].name: x})[0].reshape(5, -1)
    o_n = run_ncnn(det_n, x).reshape(5, -1)
    d_box = np.abs(o_ref[:4] - o_n[:4]).max(); d_sc = np.abs(o_ref[4] - o_n[4]).max()
    agg['det_raw_max'] = max(agg['det_raw_max'], d_box); agg['score_max'] = max(agg['score_max'], d_sc)
    dr, dn = decode(o_ref), decode(o_n)
    print(f'{path.rsplit("/",1)[-1]}: raw max|dbox| {d_box:.3f} px (model space), max|dscore| {d_sc:.4f}; dets ref {len(dr)} ncnn {len(dn)}')
    for (br, sr) in dr:
        # match by IoU to ncnn
        best = None
        for (bn, sn) in dn:
            ix = max(0, min(br[2], bn[2]) - max(br[0], bn[0])); iy = max(0, min(br[3], bn[3]) - max(br[1], bn[1]))
            i = ix * iy; u = (br[2]-br[0])*(br[3]-br[1]) + (bn[2]-bn[0])*(bn[3]-bn[1]) - i
            if i / u > 0.5 and (best is None or i / u > best[0]): best = (i / u, bn, sn)
        agg['n_ref'] += 1
        # classifier on the ref box (frame px)
        fx0, fy0, fx1, fy1 = (br[0]-left)/r, (br[1]-top)/r, (br[2]-left)/r, (br[3]-top)/r
        fx0, fy0 = max(0, fx0), max(0, fy0); fx1, fy1 = min(im.width, fx1), min(im.height, fy1)
        w, h = fx1 - fx0, fy1 - fy0
        s = 64 / min(w, h); rw, rh = max(64, round(w * s)), max(64, round(h * s))
        crop = im.crop((fx0, fy0, fx1, fy1)).resize((rw, rh), Image.BILINEAR)
        l, t = (rw - 64) // 2, (rh - 64) // 2
        c = np.asarray(crop.crop((l, t, l + 64, t + 64)), dtype=np.float32).transpose(2, 0, 1)[None] / 255
        p_ref = cls_o.run(None, {cls_o.get_inputs()[0].name: c})[0].ravel(); p_n = run_ncnn(cls_n, c).ravel()
        agg['cls_n'] += 1; agg['cls_top1_same'] += int(p_ref.argmax() == p_n.argmax()); agg['cls_prob_max'] = max(agg['cls_prob_max'], np.abs(p_ref - p_n).max())
        if best:
            agg['n_match'] += 1; agg['box_max'] = max(agg['box_max'], np.abs(br - best[1]).max())
            print(f'   ref score {sr:.3f} ncnn {best[2]:.3f} IoU {best[0]:.3f} box d {np.abs(br-best[1]).max():.2f} px | cls top1 {p_ref.argmax()} p {p_ref.max():.3f} vs ncnn {p_n.argmax()} p {p_n.max():.3f} max|dp| {np.abs(p_ref-p_n).max():.4f}')
        else:
            print(f'   ref score {sr:.3f} box {br.round(1)}: NO ncnn match')
    for (bn, sn) in dn:
        if not any(min(bn[2], br[2]) > max(bn[0], br[0]) and min(bn[3], br[3]) > max(bn[1], br[1]) for br, _ in dr):
            print(f'   ncnn-only detection score {sn:.3f} box {bn.round(1)}')
print('SUMMARY', mode, {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in agg.items()})
