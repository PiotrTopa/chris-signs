#!/usr/bin/env python3
"""Train the round-red sign verifier (signd --verifier): real sign vs lamp / traffic light / other, 32x32 RGB.

Input: an .npz from tools/verifier/build_sets.py (X uint8 [N,32,32,3] = detector box resampled to 32x32,
Y 1 = real sign, SRC source tag, SPL 'train'/'test'). Trains on SPL == 'train' only, reports per-source
accuracy on 'test', exports ONNX: input "in" [1,3,32,32] float RGB /255 (NCHW), output "logit" [1,1].
The data is research / non-commercial licensed (BDD100K, GTSRB, the CC BY 3.0 ride), so the weights
are never committed (docs/SIGN_MODELS.md "Verifier").

usage: train_verifier.py SETS.npz OUT.onnx [--epochs 12] [--threads 4] [--exclude-src PREFIX,...] [--seed 0]
"""
import argparse, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('sets'); ap.add_argument('out')
ap.add_argument('--epochs', type=int, default=12); ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--exclude-src', default=''); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed)
d = np.load(a.sets)
X, Y, SRC, SPL = d['X'], d['Y'].astype(np.float32), d['SRC'], d['SPL']
excl = [p for p in a.exclude_src.split(',') if p]
tr = np.array([s == 'train' and not any(src.startswith(p) for p in excl) for s, src in zip(SPL, SRC)])
te = SPL == 'test'


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        c = [3, 16, 32, 64]
        self.convs = nn.ModuleList([nn.Conv2d(c[i], c[i + 1], 3, padding=1) for i in range(3)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(c[i + 1]) for i in range(3)])
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        for cv, bn in zip(self.convs, self.bns):
            x = F.max_pool2d(F.relu(bn(cv(x))), 2)
        return self.fc(x.mean((2, 3)))


def augment(xb):
    """xb uint8 [B,32,32,3] -> float [B,3,32,32]: box jitter, small-box blur, photometric jitter, flips."""
    x = torch.from_numpy(xb).float().permute(0, 3, 1, 2) / 255
    B = x.shape[0]
    # box jitter: scale 0.85-1.2, shift +-10 % (detector boxes are loose / tight)
    s = torch.empty(B).uniform_(0.85, 1.2); tx = torch.empty(B).uniform_(-0.1, 0.1); ty = torch.empty(B).uniform_(-0.1, 0.1)
    th = torch.zeros(B, 2, 3); th[:, 0, 0] = s; th[:, 1, 1] = s * torch.empty(B).uniform_(0.9, 1.1); th[:, 0, 2] = tx; th[:, 1, 2] = ty
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), padding_mode='border', align_corners=False)
    # small boxes: down to 8-32 px and back up
    for i in range(B):
        if np.random.rand() < 0.6:
            n = int(np.random.randint(8, 32))
            x[i:i + 1] = F.interpolate(F.interpolate(x[i:i + 1], size=(n, n), mode='bilinear', antialias=True, align_corners=False),
                                       size=(32, 32), mode='bilinear', align_corners=False)
    # photometric: gain, gamma, per-channel gain, saturation, noise
    g = torch.empty(B, 1, 1, 1).uniform_(0.6, 1.5); x = x * g
    x = x.clamp(0, 1) ** torch.empty(B, 1, 1, 1).uniform_(0.7, 1.4)
    x = x * torch.empty(B, 3, 1, 1).uniform_(0.9, 1.1)
    m = x.mean(1, keepdim=True); x = m + (x - m) * torch.empty(B, 1, 1, 1).uniform_(0.6, 1.3)
    x = x + 0.02 * torch.randn_like(x)
    flip = torch.rand(B) < 0.5; x[flip] = x[flip].flip(3)
    return x.clamp(0, 1)


net = Net()
opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
itr = np.nonzero(tr)[0]
# balance: real and not-real equally likely per batch; sources weighted by 1/sqrt(count) inside a class
cnt = collections.Counter(SRC[itr])
w = np.array([1 / np.sqrt(cnt[SRC[i]]) for i in itr]); 
for yv in (0, 1):
    m = Y[itr] == yv; w[m] /= w[m].sum()
w /= w.sum()
steps = max(1, len(itr) // 128)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=a.epochs * steps)
for ep in range(a.epochs):
    net.train(); tot = 0
    for _ in range(steps):
        b = np.random.choice(itr, 128, p=w)
        xb = augment(X[b]); yb = torch.from_numpy(Y[b])[:, None]
        loss = F.binary_cross_entropy_with_logits(net(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
    print(f'epoch {ep} loss {tot / steps:.4f}', flush=True)

net.eval()
def predict(idx):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 512):
            x = torch.from_numpy(X[idx[i:i + 512]]).float().permute(0, 3, 1, 2) / 255
            out.append(torch.sigmoid(net(x))[:, 0].numpy())
    return np.concatenate(out) if out else np.zeros(0)
ite = np.nonzero(te)[0]; p = predict(ite)
print('held-out (SPL test), share predicted real (p >= 0.5):')
by = collections.defaultdict(list)
for i, pi in zip(ite, p):
    s = SRC[i]; by['gtsrb' if s.startswith('gtsrb') else s].append(pi)
for k in sorted(by):
    v = np.array(by[k]); print(f'  {k:26s} n={len(v):5d} real-rate {np.mean(v >= 0.5):.3f}  (label {"real" if k.startswith(("gtsrb", "krakow_real", "bdd_sign")) else "not real"})')
np.save(a.out + '.test_pred.npy', np.stack([ite, p]))
torch.onnx.export(net, torch.zeros(1, 3, 32, 32), a.out, input_names=['in'], output_names=['logit'], opset_version=13, dynamo=False)
print('wrote', a.out)
