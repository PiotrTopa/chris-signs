#!/usr/bin/env python3
"""Train the traffic-light state CNN (same network, input and classes as train_state.py: 1x3x48x24, logits
red / yellow / green / none) with augmentations that reproduce what a dashcam behind a windscreen does to a
small lamp and what a detector does to the box, which the 2026-10-27 model never saw:

  * box jitter: the crop is cut from the image with the box scaled by 0.85-1.25 and shifted by up to 12 % of
    its size (a detector's box is not the annotator's);
  * size: the crop is reduced to a height between 7 px and its own and enlarged again (far lights are
    7-20 px on a 720p camera; BDD boxes of that size are rare in the training half);
  * veiling glare: x * (1 - a) + a * c with a up to 0.65 and c a bright, slightly warm or cold grey (sun in or
    near the picture lifts the black housing and washes the lamp out);
  * clipped highlights (--whiteout P): every pixel brighter than a threshold is pushed towards white, the more
    the brighter. Against a bright sky the car's camera shows a lit lamp as a white disc with a thin
    coloured rim on a dark housing; only its place in the housing and the rim say which lamp it is;
  * exposure / gamma / per-channel gain, blur, JPEG at quality 25-90 at the reduced size (MJPEG cameras).

Same data and split as train_state.py (BDD100K val, images split by sha1 of the name: even = train, odd =
test); the test half is evaluated without augmentation, so the numbers compare with the earlier model.
BDD100K is research / non-commercial: weights are local, never committed (docs/LIGHT_MODELS.md).

    python3 tools/lights/train_state2.py SAMPLES.json IMAGE_ROOT OUT_STEM [--epochs 60] [--width 1.0] [--aug 1] [--whiteout 0.5]
"""
import argparse, collections, hashlib, json, os
import numpy as np, cv2, torch, torch.nn as nn

H, W = 48, 24
WHITEOUT = 0.0
CL = {"R": 0, "Y": 1, "G": 2, "NA": 3}

def split(name):
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) % 2

def load(samples, root, min_h=8, margin=0.35):
    """-> list of (context crop RGB uint8, box inside it x0,y0,x1,y1, label, split)."""
    d = json.load(open(samples)); out = []
    for s in d["samples"]:
        iw, ih = s["metadata"]["width"], s["metadata"]["height"]
        tls = [t for t in (s.get("detections") or {}).get("detections") or [] if t["label"] == "traffic light"]
        img = None
        for t in tls:
            x, y, w, h = t["bounding_box"]
            x0, y0, x1, y1 = x * iw, y * ih, (x + w) * iw, (y + h) * ih
            if y1 - y0 < min_h or x1 - x0 < 3 or t.get("trafficLightColor") not in CL: continue
            if img is None: img = cv2.cvtColor(cv2.imread(os.path.join(root, s["filepath"])), cv2.COLOR_BGR2RGB)
            mx, my = margin * (x1 - x0), margin * (y1 - y0)
            X0, Y0, X1, Y1 = int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(iw, np.ceil(x1 + mx))), int(min(ih, np.ceil(y1 + my)))
            out.append((img[Y0:Y1, X0:X1].copy(), (x0 - X0, y0 - Y0, x1 - X0, y1 - Y0), CL[t["trafficLightColor"]], split(s["filepath"])))
    return out

def cut(ctx, box, rng=None, aug=1.0):
    x0, y0, x1, y1 = box; w, h = x1 - x0, y1 - y0
    if rng is not None:
        sc = rng.uniform(1 - 0.15 * aug, 1 + 0.25 * aug, 2); sh = rng.uniform(-0.12 * aug, 0.12 * aug, 2)
        cx, cy = (x0 + x1) / 2 + sh[0] * w, (y0 + y1) / 2 + sh[1] * h; w, h = w * sc[0], h * sc[1]
        x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    X0, Y0 = int(np.clip(round(x0), 0, ctx.shape[1] - 2)), int(np.clip(round(y0), 0, ctx.shape[0] - 3))
    X1, Y1 = int(np.clip(round(x1), X0 + 2, ctx.shape[1])), int(np.clip(round(y1), Y0 + 3, ctx.shape[0]))
    c = ctx[Y0:Y1, X0:X1]
    if (x1 - x0) > 1.2 * (y1 - y0): c = cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE)
    return c

def degrade(c, rng, aug=1.0):
    h, w = c.shape[:2]
    if rng.random() < 0.8 * aug:                                   # size
        nh = int(rng.uniform(7, max(8, h))); nw = max(3, round(w * nh / h))
        c = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.35 * aug: c = cv2.GaussianBlur(c, (3, 3), rng.uniform(0.4, 1.0))
    x = c.astype(np.float32) / 255
    if rng.random() < WHITEOUT * aug:                              # clipped highlights
        v = x.max(2, keepdims=True); t = rng.uniform(0.3, 0.6)
        m = np.clip((v - t) / rng.uniform(0.15, 0.35), 0, 1); m = m * m * (3 - 2 * m)
        x = x + rng.uniform(0.4, 1.0) * m * (1 - x)
    if rng.random() < 0.5 * aug:                                   # veiling glare
        a = rng.uniform(0.05, 0.65); col = rng.uniform(0.75, 1.0) * np.array([1, rng.uniform(0.93, 1.03), rng.uniform(0.82, 1.08)], np.float32)
        x = x * (1 - a) + a * col
    x = x * rng.uniform(0.55, 1.5)                                 # exposure
    if rng.random() < 0.5 * aug: x = np.clip(x, 0, 1) ** rng.uniform(0.6, 1.6)
    if rng.random() < 0.5 * aug: x = x * rng.uniform(0.9, 1.1, 3).astype(np.float32)
    c = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    if rng.random() < 0.6 * aug:                                   # MJPEG
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(c, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(25, 90))])
        c = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return c

def to_input(c):
    return cv2.resize(c, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32).transpose(2, 0, 1) / 255.0

class Net(nn.Module):
    def __init__(self, width=1.0):
        super().__init__()
        c = [max(8, int(round(v * width))) for v in (16, 32, 64, 64)]
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        self.f = nn.Sequential(blk(3, c[0]), nn.MaxPool2d(2), blk(c[0], c[1]), nn.MaxPool2d(2), blk(c[1], c[2]), nn.MaxPool2d(2), blk(c[2], c[3]))
        self.h = nn.Linear(c[3] * 6 * 3, 4)
    def forward(self, x): return self.h(self.f(x).flatten(1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples"), ap.add_argument("root"), ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--wpow", type=float, default=0.85)
    ap.add_argument("--width", type=float, default=1.0); ap.add_argument("--aug", type=float, default=1.0)
    ap.add_argument("--whiteout", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    global WHITEOUT
    WHITEOUT = a.whiteout
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    cache = a.out + ".data.npy"
    data = load(a.samples, a.root)
    tr = [d for d in data if d[3] == 0]; te = [d for d in data if d[3] == 1]
    Ytr = np.array([d[2] for d in tr]); Yte = np.array([d[2] for d in te])
    print("train", dict(collections.Counter(Ytr.tolist())), "test", dict(collections.Counter(Yte.tolist())), flush=True)
    Xte = torch.tensor(np.stack([to_input(cut(d[0], d[1])) for d in te]))
    net = Net(a.width)
    w = torch.tensor([1.0 / np.mean(Ytr == k) for k in range(4)], dtype=torch.float32); w = (w / w.mean()) ** a.wpow
    steps = a.epochs * (len(tr) // 128 + 1)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    Yt = torch.tensor(Ytr)
    for ep in range(a.epochs):
        net.train()
        X = np.stack([to_input(degrade(cut(d[0], d[1], rng, a.aug), rng, a.aug) if a.aug > 0 else cut(d[0], d[1])) for d in tr])
        Xt = torch.tensor(X); perm = torch.randperm(len(Xt)); tot = 0.0
        for i in range(0, len(Xt), 128):
            b = perm[i:i + 128]; x = Xt[b]
            flip = torch.rand(len(b)) < 0.5; x = x.clone(); x[flip] = x[flip].flip(3)
            opt.zero_grad(); loss = lossf(net(x), Yt[b]); loss.backward(); opt.step(); sched.step(); tot += float(loss.detach()) * len(b)
        if ep % 10 == 9 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad(): p = net(Xte).argmax(1).numpy()
            print("epoch %d loss %.3f test recall red %.1f yellow %.1f green %.1f none %.1f" % ((ep + 1, tot / len(Xt)) + tuple(100 * np.mean(p[Yte == k] == k) for k in range(4))), flush=True)
    torch.save(net.state_dict(), a.out + ".pt")
    net.eval()
    torch.onnx.export(net, torch.zeros(1, 3, H, W), a.out + "_48x24.onnx", input_names=["images"], output_names=["logits"], opset_version=17, dynamo=False)
    print("wrote", a.out + "_48x24.onnx")

if __name__ == "__main__":
    main()
