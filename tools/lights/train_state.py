#!/usr/bin/env python3
"""Train the traffic-light state CNN that `signd --lights-state PATH.onnx` runs on each light crop.

Input 1x3x48x24 (H x W) RGB / 255, the box resampled without margin; housings wider than 1.2x their height are
rotated clockwise first (left lamp on top). Output 4 logits: red, yellow, green, none (BDD100K "NA": dark,
side or back view, unclear). ~0.1 MFLOP-scale net, ~1 ms per crop on ORT (one A75 thread).

Data: BDD100K 2D detection labels with the trafficLightColor attribute (G / Y / R / NA), here the 10 000 val
images as exported by FiftyOne (samples.json with relative boxes; e.g. the HF mirror dgural/bdd100k). Images are
split by the sha1 of the file name (even = train, odd = test), boxes >= 10 px tall. BDD100K is licensed for
research / non-commercial use: the trained weights are for private use only, never commit or re-host them
(docs/LIGHT_MODELS.md).

    python3 tools/lights/train_state.py SAMPLES.json IMAGE_ROOT OUT_STEM [--epochs 40] [--wpow 0.85]
    -> OUT_STEM.pt, OUT_STEM_48x24.onnx, test recall per class on stdout
Needs torch, numpy, pillow, onnx (any recent versions; tested torch 2.14 CPU).
Result 2026-10-27 (roadsense docs/lanes/L7-traffic-lights.md): test recall red 87.1, yellow 61.0, green 95.8, none 88.2 %.
"""
import argparse
import collections
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

H, W = 48, 24
CL = {"R": 0, "Y": 1, "G": 2, "NA": 3}


def split(name):
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) % 2


def load_crops(samples, root, min_h=10):
    d = json.load(open(samples))
    X, Y, S = [], [], []
    for s in d["samples"]:
        iw, ih = s["metadata"]["width"], s["metadata"]["height"]
        tls = [t for t in (s.get("detections") or {}).get("detections") or [] if t["label"] == "traffic light"]
        img = None
        for t in tls:
            x, y, w, h = t["bounding_box"]
            x0, y0, x1, y1 = x * iw, y * ih, (x + w) * iw, (y + h) * ih
            if y1 - y0 < min_h or t.get("trafficLightColor") not in CL:
                continue
            if img is None:
                img = np.asarray(Image.open(os.path.join(root, s["filepath"])).convert("RGB"))
            c = img[int(max(0, y0)):int(np.ceil(y1)), int(max(0, x0)):int(np.ceil(x1))]
            if c.size == 0:
                continue
            if c.shape[1] > 1.2 * c.shape[0]:
                c = np.rot90(c, -1)
            X.append(np.asarray(Image.fromarray(np.ascontiguousarray(c)).resize((W, H), Image.BILINEAR), np.float32)
                     .transpose(2, 0, 1) / 255.0)
            Y.append(CL[t["trafficLightColor"]])
            S.append(split(s["filepath"]))
    return np.stack(X), np.array(Y), np.array(S)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        self.f = nn.Sequential(blk(3, 16), nn.MaxPool2d(2), blk(16, 32), nn.MaxPool2d(2), blk(32, 64), nn.MaxPool2d(2),
                               blk(64, 64))
        self.h = nn.Linear(64 * 6 * 3, 4)

    def forward(self, x):
        return self.h(self.f(x).flatten(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples"), ap.add_argument("root"), ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--wpow", type=float, default=0.85, help="class weight = (1/frequency)^wpow")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    X, Y, S = load_crops(a.samples, a.root)
    tr, te = S == 0, S == 1
    print("train", dict(collections.Counter(Y[tr])), "test", dict(collections.Counter(Y[te])))
    net = Net()
    Xt, Yt = torch.tensor(X[tr]), torch.tensor(Y[tr])
    w = torch.tensor([1.0 / np.mean(Y[tr] == k) for k in range(4)], dtype=torch.float32)
    w = (w / w.mean()) ** a.wpow
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=a.epochs * (len(Xt) // 128 + 1))
    lossf = nn.CrossEntropyLoss(weight=w)
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            b = perm[i:i + 128]
            x = Xt[b].clone() * (0.6 + 0.8 * torch.rand(len(b), 1, 1, 1))     # brightness
            flip = torch.rand(len(b)) < 0.5
            x[flip] = x[flip].flip(3)                                         # mirror left/right
            x = torch.roll(x, (np.random.randint(-3, 4), np.random.randint(-2, 3)), (2, 3)).clamp(0, 1)
            opt.zero_grad()
            loss = lossf(net(x), Yt[b])
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    with torch.no_grad():
        p = net(torch.tensor(X[te])).argmax(1).numpy()
    rec = [100 * np.mean(p[Y[te] == k] == k) for k in range(4)]
    print("test recall red %.1f yellow %.1f green %.1f none %.1f" % tuple(rec))
    torch.save(net.state_dict(), a.out + ".pt")
    torch.onnx.export(net, torch.zeros(1, 3, H, W), a.out + "_48x24.onnx", input_names=["images"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    print("wrote", a.out + "_48x24.onnx")


if __name__ == "__main__":
    main()
