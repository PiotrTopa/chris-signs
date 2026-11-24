#!/usr/bin/env python3
"""train_reid.py OUT_STEM RIDE.npz...: a small re-identification network on ride crops, no labels: identities are
the dense IoU chains of each ride (reid_lib.chains; chains of at least four boxes over half a second that no
other box covers by 0.3 or more), positives the crops of one chain.
Network: 32x32 RGB -> 4 conv blocks (16, 32, 64, 96) -> global mean -> 32 numbers, L2-normalised (214 k
parameters). Loss: supervised contrastive over batches of P chains x K crops. Augmentation: shift / scale jitter
of the crop, brightness, contrast, gamma, channel gain, blur, noise. Writes OUT_STEM.pt, OUT_STEM.onnx (input
1x3x32x32 RGB / 255, for signd --lights-ru-emb) and OUT_STEM-batch.onnx (any batch, for reid_eval.py)."""
import os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reid_lib as L
from reid_net import Net, export

stem, files = sys.argv[1], [a for a in sys.argv[2:] if a.endswith('.npz')]
iters = int(os.environ.get('ITERS', 4000)); P, K = 48, 4
torch.manual_seed(1); rng = np.random.default_rng(1)
torch.set_num_threads(int(os.environ.get('THREADS', 4)))
crops, ident = [], []
base = 0
for f in files:
    rows, c = L.load(f)
    tid = L.chains(rows)
    fr = rows[:, 0]
    keep = (rows[:, 7] < 0.3)  # not much covered by another box
    for t in np.unique(tid[keep]):
        m = np.where((tid == t) & keep)[0]
        if len(m) < 4 or fr[m].max() - fr[m].min() < 10: continue  # at least four crops over half a second
        crops.append(c[m]); ident.append(np.full(len(m), base, int)); base += 1
crops = np.concatenate(crops); ident = np.concatenate(ident)
by = [np.where(ident == i)[0] for i in range(base)]
print('%d crops of %d chains from %d rides' % (len(crops), base, len(files)), flush=True)

def augment(x):  # x [n,3,32,32] RGB 0..1
    n = x.shape[0]
    th = torch.zeros(n, 2, 3)
    sc = 1 + (torch.rand(n) - 0.5) * 0.24
    th[:, 0, 0] = sc; th[:, 1, 1] = sc * (1 + (torch.rand(n) - 0.5) * 0.1)
    th[:, :, 2] = (torch.rand(n, 2) - 0.5) * 0.24
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), padding_mode='border', align_corners=False)
    g = torch.exp((torch.rand(n, 1, 1, 1) - 0.5) * 0.8)                      # gamma
    x = x.clamp(1e-4, 1) ** g
    x = x * (1 + (torch.rand(n, 1, 1, 1) - 0.5) * 0.6) + (torch.rand(n, 1, 1, 1) - 0.5) * 0.15   # contrast, brightness
    x = x * (1 + (torch.rand(n, 3, 1, 1) - 0.5) * 0.12)                       # channel gain
    blur = torch.rand(n) < 0.3
    if blur.any(): x[blur] = F.avg_pool2d(F.pad(x[blur], (1, 1, 1, 1), mode='replicate'), 3, 1)
    return (x + torch.randn_like(x) * 0.02).clamp(0, 1)

net = Net(); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=iters)
T = 0.1
for it in range(iters):
    ids = rng.choice(base, P, replace=False)
    sel = np.concatenate([rng.choice(by[i], K, replace=len(by[i]) < K) for i in ids])
    x = torch.from_numpy(crops[sel][..., ::-1].copy()).permute(0, 3, 1, 2).float() / 255
    y = torch.from_numpy(np.repeat(np.arange(P), K))
    z = net(augment(x))
    sim = z @ z.t() / T
    eye = torch.eye(len(y), dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    loss = -(logp * pos).sum(1).div(pos.sum(1).clamp(min=1)).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it % 200 == 0 or it == iters - 1: print(it, round(float(loss), 4), flush=True)
net.eval()
torch.save(net.state_dict(), stem + '.pt')
export(net, stem)
print('parameters', sum(p.numel() for p in net.parameters()))
