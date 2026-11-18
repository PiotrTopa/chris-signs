#!/usr/bin/env python3
"""Fine-tune the traffic-light state CNN on labelled detections from the car's own camera, next to the BDD100K
crops it was trained on, and say honestly what that buys: grouped cross-validation over the car sample (folds
are one-minute blocks of a drive, so the frames of one approach to one light never sit on both sides), then
one model on all of it.

    finetune_state_car.py BASE.pt SAMPLES.json IMAGE_ROOT SCAN.npz SHEETS_DIR OUT_STEM [--epochs 12] [--share 0.3]

BASE.pt: train_state.py / train_state2.py weights (same network). SCAN.npz + SHEETS_DIR: car_scan.py output and
the labelled sample (car_sheets.py, labels.txt). Car labels r / y / g train their class, `ry` trains red (the
lit red lamp is what the crop shows; the tracker makes red+yellow from the sequence), x / o train `none`.
Every batch holds `--share` car crops (jittered box, mirror, exposure), the rest BDD crops (exposure, mirror).
Prints per fold and pooled: lit lights given the right colour / a wrong colour / none, others called lit
(weighted by the sample's strata), and the BDD test recalls of every model. Weights are local only (BDD100K
is research-licensed; the car frames are the owner's)."""
import argparse, json, os, sys
import numpy as np, cv2, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_state2 as T

def car_data(scan, sheets):
    z = np.load(scan, allow_pickle=True); meta = json.loads(str(z["meta"])); crops = z["crops"]
    S = json.load(open(os.path.join(sheets, "sample.json")))
    lab = dict((int(a), b) for a, b in (l.split() for l in open(os.path.join(sheets, "labels.txt"))))
    n_none = S["strata"]["cnn_none"][1]; out = []
    for k, i in enumerate(S["sample"]):
        m = meta[i]; X0, Y0 = m["ctx"][0], m["ctx"][1]; b = m["box"]
        w = S["strata"]["cnn_none"][0] / n_none if k < n_none else S["strata"]["cnn_lit"][0] / (len(S["sample"]) - n_none)
        out.append(dict(ctx=crops[i], box=(b[0] - X0, b[1] - Y0, b[2] - X0, b[3] - Y0), label=lab[k], w=w,
                        group="%s/%d" % (m["session"], m["seq"] // 1200)))
    return out

CLS = {"r": 0, "ry": 0, "y": 1, "g": 2, "x": 3, "o": 3}
NAMES = ["r", "y", "g", "none"]

def car_input(d, rng=None):
    c = T.cut(d["ctx"], d["box"], rng, 0.8)
    if rng is not None:
        x = np.clip(c.astype(np.float32) * rng.uniform(0.75, 1.25), 0, 255).astype(np.uint8)
        return T.to_input(x)
    return T.to_input(c)

def metrics(net, car):
    net.eval()
    with torch.no_grad(): p = net(torch.tensor(np.stack([car_input(d) for d in car]))).argmax(1).numpy()
    lit = {"right": 0.0, "wrong": 0.0, "none": 0.0}; oth = {"lit": 0.0, "none": 0.0}
    for d, k in zip(car, p):
        l, pn = d["label"], NAMES[k]
        if l in ("r", "y", "g", "ry"):
            ok = pn == l or (l == "ry" and pn in ("r", "y"))
            lit["right" if ok else "none" if pn == "none" else "wrong"] += d["w"]
        else: oth["lit" if pn != "none" else "none"] += d["w"]
    return lit, oth

def fmt(lit, oth):
    tl, to = sum(lit.values()) or 1, sum(oth.values()) or 1
    return "lit lights: right %5.1f %%  wrong colour %4.1f %%  none %5.1f %%  | others called lit %5.1f %%" % (
        100 * lit["right"] / tl, 100 * lit["wrong"] / tl, 100 * lit["none"] / tl, 100 * oth["lit"] / to)

def train(base, Xb, Yb, car_tr, epochs, share, rng, wpow=0.85):
    net = T.Net(); net.load_state_dict(torch.load(base, map_location="cpu")); net.train()
    w = torch.tensor([1.0 / max(1e-6, np.mean(Yb == k)) for k in range(4)], dtype=torch.float32); w = (w / w.mean()) ** wpow
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(net.parameters(), 4e-4, weight_decay=1e-4)
    nb = 128; nc = int(round(nb * share)) if car_tr else 0; steps = len(Xb) // (nb - nc)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps)
    Xt, Yt = torch.tensor(Xb), torch.tensor(Yb)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for s in range(steps):
            b = perm[s * (nb - nc):(s + 1) * (nb - nc)]
            x = Xt[b] * (0.6 + 0.8 * torch.rand(len(b), 1, 1, 1)); y = Yt[b]
            if nc:
                idx = rng.integers(0, len(car_tr), nc)
                x = torch.cat([x, torch.tensor(np.stack([car_input(car_tr[i], rng) for i in idx]))])
                y = torch.cat([y, torch.tensor([CLS[car_tr[i]["label"]] for i in idx])])
            flip = torch.rand(len(x)) < 0.5; x = x.clone(); x[flip] = x[flip].flip(3)
            opt.zero_grad(); lossf(net(x.clamp(0, 1)), y).backward(); opt.step(); sched.step()
    return net

def bdd_recall(net, Xte, Yte):
    net.eval()
    with torch.no_grad(): p = net(torch.tensor(Xte)).argmax(1).numpy()
    return "BDD test recall red %.1f yellow %.1f green %.1f none %.1f" % tuple(100 * np.mean(p[Yte == k] == k) for k in range(4))

def main():
    ap = argparse.ArgumentParser()
    for a in ("base", "samples", "root", "scan", "sheets", "out"): ap.add_argument(a)
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--share", type=float, default=0.3)
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(0); rng = np.random.default_rng(0)
    cache = a.out + ".bdd.npz"
    if os.path.exists(cache):
        z = np.load(cache); Xb, Yb, Xte, Yte = z["Xb"], z["Yb"], z["Xte"], z["Yte"]
    else:
        data = T.load(a.samples, a.root)
        tr = [d for d in data if d[3] == 0]; te = [d for d in data if d[3] == 1]
        Xb = np.stack([T.to_input(T.cut(d[0], d[1])) for d in tr]); Yb = np.array([d[2] for d in tr])
        Xte = np.stack([T.to_input(T.cut(d[0], d[1])) for d in te]); Yte = np.array([d[2] for d in te])
        np.savez(cache, Xb=Xb, Yb=Yb, Xte=Xte, Yte=Yte)
    car = car_data(a.scan, a.sheets)
    base = T.Net(); base.load_state_dict(torch.load(a.base, map_location="cpu"))
    print("base           car:", fmt(*metrics(base, car)), "|", bdd_recall(base, Xte, Yte), flush=True)
    groups = sorted(set(d["group"] for d in car)); rng.shuffle(groups)
    pl = {"right": 0.0, "wrong": 0.0, "none": 0.0}; po = {"lit": 0.0, "none": 0.0}
    for f in range(a.folds):
        held = set(groups[f::a.folds])
        tr = [d for d in car if d["group"] not in held]; te = [d for d in car if d["group"] in held]
        net = train(a.base, Xb, Yb, tr, a.epochs, a.share, rng)
        lit, oth = metrics(net, te)
        for k in pl: pl[k] += lit[k]
        for k in po: po[k] += oth[k]
        print("fold %d (%3d held out) car:" % (f, len(te)), fmt(lit, oth), "|", bdd_recall(net, Xte, Yte), flush=True)
    print("cross-validated car:", fmt(pl, po), flush=True)
    ctrl = train(a.base, Xb, Yb, [], a.epochs, 0.0, rng)
    print("control (same fine-tuning, no car crops) car:", fmt(*metrics(ctrl, car)), "|", bdd_recall(ctrl, Xte, Yte), flush=True)
    net = train(a.base, Xb, Yb, car, a.epochs, a.share, rng)
    print("all car labels (in-sample) car:", fmt(*metrics(net, car)), "|", bdd_recall(net, Xte, Yte), flush=True)
    torch.save(net.state_dict(), a.out + ".pt"); net.eval()
    torch.onnx.export(net, torch.zeros(1, 3, T.H, T.W), a.out + "_48x24.onnx", input_names=["images"], output_names=["logits"], opset_version=17, dynamo=False)
    print("wrote", a.out + "_48x24.onnx")

if __name__ == "__main__":
    main()
