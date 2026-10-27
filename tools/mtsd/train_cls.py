#!/usr/bin/env python3
"""Train the 64x64 sign classifier (YOLOv8n-cls architecture, Ultralytics, from scratch) on MTSD crops plus
`other-sign` and `background` (not a sign) classes, and export it to ONNX like the original
(input `images` [1,3,64,64] RGB /255, output `output0` [1,K] softmax).

usage: train_cls.py --classes BASE_CLASSES.txt --train DIR [DIR ...] --out OUT_DIR [--epochs 30] [--bdd BDD.npz]

Class list: the 400 names of BASE_CLASSES.txt in the same order (ids unchanged), then `other-sign`, `background`.
Sample selection (from extract_crops.py / extract_bdd.py shards): gt + detpos of every class except `ambiguous`
objects; `other-sign` gt kept at 50 %; detneg all; rand 50 %; BDD background crops of the train split only.
Rendering: cls_aug.train_sample (box jitter, phone-size downscale, JPEG, NV12 chroma, photometric). No flips.
Sampling: per class weight 1 / sqrt(count) (long tail), CE with label smoothing 0.1, AdamW, cosine schedule.
MTSD (CC BY-NC-SA 4.0) and BDD100K (non-commercial research) derived weights are never committed.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cls_aug  # noqa: E402


def load(dirs, names, bdd=None, seed=0, split=None):
    idx = {n: i for i, n in enumerate(names)}
    rng = np.random.default_rng(seed)
    Xs, Y, MM = [], [], []
    files = sorted(f for d in dirs for f in glob.glob(f'{d}/*.npz'))
    if bdd: files.append(bdd)
    for f in files:
        d = np.load(f); M = json.loads(str(d['meta']))
        keep = []
        for i, m in enumerate(M):
            if split and m.get('split', split) != split: continue
            lab = m['label']; k = m['kind']
            if lab not in idx: continue
            if 'ambiguous' in m.get('flags', []) or 'dummy' in m.get('flags', []): continue
            if k == 'gt' and lab == 'other-sign' and rng.random() > 0.5: continue
            if k == 'rand' and rng.random() > 0.5: continue
            keep.append(i)
        if not keep: continue
        X = d['X']
        Xs.append(X[keep]); Y += [idx[M[i]['label']] for i in keep]
        MM += [{'sw': M[i]['sw'], 'sh': M[i]['sh'], 'w': M[i]['w'], 'h': M[i]['h'], 'W': M[i]['W'], 'kind': M[i]['kind']} for i in keep]
        print(f'{os.path.basename(f)}: {len(keep)} of {len(M)}', flush=True)
    return np.concatenate(Xs), np.array(Y, np.int64), MM


class DS(torch.utils.data.Dataset):
    def __init__(self, X, Y, M): self.X, self.Y, self.M = X, Y, M
    def __len__(self): return len(self.Y)
    def __getitem__(self, i):
        rng = np.random.default_rng((i * 7919 + int(time.time_ns() % 1000003)) & 0xffffffff)
        im = cls_aug.train_sample(self.X[i], self.M[i], rng)
        return torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))), int(self.Y[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', required=True); ap.add_argument('--train', nargs='+', required=True)
    ap.add_argument('--bdd'); ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--epoch-size', type=int, default=200000)
    ap.add_argument('--batch', type=int, default=256); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = [l.strip() for l in open(a.classes) if l.strip()]
    names = base + ['other-sign', 'background']
    open(f'{a.out}/classes.txt', 'w').write('\n'.join(names) + '\n')
    X, Y, M = load(a.train, names, a.bdd, split='train')
    cnt = np.bincount(Y, minlength=len(names))
    print('samples', len(Y), 'classes with data', int((cnt > 0).sum()), 'other-sign', cnt[-2], 'background', cnt[-1], flush=True)
    w = np.where(cnt > 0, 1.0 / np.sqrt(np.maximum(cnt, 1)), 0)[Y]
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w, dtype=torch.double), a.epoch_size, replacement=True)
    dl = torch.utils.data.DataLoader(DS(X, Y, M), batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                                     persistent_workers=True, drop_last=True, prefetch_factor=4)
    from ultralytics.nn.tasks import ClassificationModel
    torch.manual_seed(0)
    model = ClassificationModel('yolov8n-cls.yaml', nc=len(names), verbose=False).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * (a.epoch_size // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05, anneal_strategy='cos')
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    log = open(f'{a.out}/train.log', 'a')
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tl = tc = tn = 0
        for xb, yb in dl:
            xb = xb.cuda(non_blocking=True).float() / 255.0; yb = yb.cuda(non_blocking=True)
            out = model(xb)
            loss = lossf(out, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(yb); tc += (out.argmax(1) == yb).sum().item(); tn += len(yb)
        msg = f'epoch {ep + 1}/{a.epochs} loss {tl / tn:.4f} acc {tc / tn:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s'
        print(msg, flush=True); log.write(msg + '\n'); log.flush()
        torch.save({'model': model.state_dict(), 'names': names, 'epoch': ep + 1}, f'{a.out}/last.pth')
    export(model, names, f'{a.out}/signcls_64.onnx')


def export(model, names, path):
    model = model.cpu().eval()
    head = model.model[-1]; head.export = True
    x = torch.zeros(1, 3, 64, 64)
    torch.onnx.export(model, x, path, input_names=['images'], output_names=['output0'], opset_version=17,
                      dynamo=False, do_constant_folding=True)
    import onnx
    m = onnx.load(path)
    m.metadata_props.add(key='names', value=json.dumps({i: n for i, n in enumerate(names)}))
    onnx.save(m, path)
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    xr = np.random.default_rng(0).random((1, 3, 64, 64), np.float32)
    with torch.no_grad(): ref = model(torch.from_numpy(xr)).numpy()
    got = s.run(None, {'images': xr})[0]
    print('export', path, 'shape', got.shape, 'max |onnx - torch|', float(np.abs(got - ref).max()), 'sum', float(got.sum()))


if __name__ == '__main__':
    main()
