#!/usr/bin/env python3
"""reid_eval.py [--onnx MODEL.onnx] [--dist OUT.json] RIDE.npz...: re-identification value of appearance vectors on
ride crops (crops_extract.py). Identities: dense IoU chains. Score per ride and gap (1, 2, 4 s at the unit's 5 Hz):
"frame": the query's own later box against the other boxes of that frame (the association a tracker faces; two
boxes of one frame are two objects); "ride": against every box of every other chain of the ride (harder, and it
counts another chain of the same car as a miss). Per gap: queries, rank-1 share, AUC.
--onnx: also the learnt embedding (train_reid.py; input 1x3x32x32 RGB / 255, output the vector).
--dist: writes, per ride, the same-object / other-object distance quantiles and log-normal fits of the vectors
as signd publishes them (for a tracker's likelihood ratio)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reid_lib as L

args = sys.argv[1:]
onnx = args[args.index('--onnx') + 1] if '--onnx' in args else None
dist = args[args.index('--dist') + 1] if '--dist' in args else None
files = [a for i, a in enumerate(args) if a.endswith('.npz')]
sess = None
if onnx:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx, providers=['CPUExecutionProvider'])

def learnt(crops):
    x = crops[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32) / 255
    out = []
    for i in range(0, len(x), 512):
        out.append(sess.run(None, {sess.get_inputs()[0].name: x[i:i + 512]})[0])
    return np.concatenate(out)

def q8(x, scale):  # int8 as published
    return np.clip(np.round(x * scale), -127, 127) / scale

DESCS = {
    'yuv 2x2 (12)': lambda c: L.desc_yuv_grid(c, 2),
    'yuv 3x3 (27)': lambda c: L.desc_yuv_grid(c, 3),
    'yuv 4x4 (48)': lambda c: L.desc_yuv_grid(c, 4),
    'yuv 3x3, Y de-meaned': lambda c: L.desc_yuv_grid(c, 3, demean_y=True),
    'yuv 3x3, Y x0.5': lambda c: L.desc_yuv_grid(c, 3, wy=0.5),
    'yuv 3x3, chroma x2': lambda c: L.desc_yuv_grid(c, 3, wc=2.0),
    'yuv 3x3, chroma only': lambda c: L.desc_yuv_grid(c, 3, wy=0.0),
    'yuv 3x3 int8 (x32)': lambda c: q8(L.desc_yuv_grid(c, 3), 32.0),
}
if sess:
    DESCS['learnt'] = learnt
    DESCS['learnt int8 (x127)'] = lambda c: q8(learnt(c), 127.0)
    DESCS['learnt + yuv 3x3'] = lambda c: np.concatenate([learnt(c), 0.5 * L.desc_yuv_grid(c, 3)], 1)
    # as signd publishes it (emb_kind reid32+yuv3x3): both parts int8, distance sqrt(|d learnt|^2 + (0.5 |d yuv|)^2)
    DESCS['learnt + yuv 3x3 int8'] = lambda c: np.concatenate([q8(learnt(c), 127.0), 0.5 * q8(L.desc_yuv_grid(c, 3), 32.0)], 1)
print('%-24s %s' % ('ride / vector', '   '.join('gap %d s: n rank-1 AUC' % g for g in (1, 2, 4))))
if '--train-usable' in args: pass
D = {}
for f in files:
    rows, crops = L.load(f)
    tid = L.chains(rows)
    w = rows[:, 5] - rows[:, 3]
    veh = L.group(rows[:, 1]) == 1
    nch = len(set(tid[veh]))
    print('%s: %d boxes (%d vehicles in %d chains), width p10/p50/p90 %.0f/%.0f/%.0f px' % (os.path.basename(f), len(rows), veh.sum(), nch, *np.percentile(w[veh], [10, 50, 90])))
    for name, fn in DESCS.items():
        X = fn(crops)
        resf = [L.retrieval_frame(X, rows, tid, g) for g in (1, 2, 4)]
        print('  %-22s frame %s  (others in the frame: %.1f)' % (name, '   '.join('%5d %.3f %.4f' % r[:3] for r in resf), resf[0][3]))
        if '--no-ride' not in args:
            res = [L.retrieval(X, rows, tid, g) for g in (1, 2, 4)]
            print('  %-22s ride  %s' % ('', '   '.join('%5d %.3f %.4f' % r for r in res)))
        if dist and name in ('yuv 3x3 int8 (x32)', 'learnt + yuv 3x3 int8'):
            for g in (1, 2, 4):
                p, n = L.pair_distances(X, rows, tid, g)
                rk = os.path.basename(f).replace('crops-', '').replace('.npz', '')
                D.setdefault(rk, {}).setdefault(name, {}).setdefault(str(g), {'pos': [], 'neg': []})
                D[rk][name][str(g)]['pos'] += p.tolist(); D[rk][name][str(g)]['neg'] += n.tolist()
    if '--size' in args:
        for name in ('yuv 3x3 (27)',) + (('learnt',) if sess else ()):
            X = DESCS[name](crops)
            for lo, hi in ((0, 24), (24, 48), (48, 96), (96, 1e9)):
                r = L.retrieval(X, rows, tid, 2, minw=lo, maxw=hi)
                print('    %-20s width %3g-%-4g px, gap 2 s: %5d %.3f %.4f' % (name, lo, hi, *r))
if dist:
    # per descriptor and gap: quantiles of the distance between two boxes of one object (gap apart) and between a
    # box and the OTHER boxes of the later frame; a log-normal fit of each (mu, sigma of ln d) and, with those fits
    # and a likelihood ratio clamped to [1/8, 8], the share of other-object pairs at LR <= 1/4 and of same-object
    # pairs at LR >= 4
    KIND = {'yuv 3x3 int8 (x32)': 'yuv3x3', 'learnt + yuv 3x3 int8': 'reid32+yuv3x3'}
    out = {'note': 'distance = Euclidean on the published int8 values in the units of roadsense docs/ARCHITECTURE.md '
                   '(yuv3x3: int8 / 32; reid32+yuv3x3: first 32 / 127, the other 27 / 32 x 0.5); rides: ' + ', '.join(os.path.basename(f) for f in files)}
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    def lognorm(v):
        l = np.log(np.maximum(np.asarray(v), 1e-3)); return float(l.mean()), float(l.std())
    def lpdf(d, m, sg):
        return -np.log(d * sg * np.sqrt(2 * np.pi)) - (np.log(d) - m) ** 2 / (2 * sg * sg)
    for rk, name, d in [(rk, name, d) for rk, dd in D.items() for name, d in dd.items()]:
        o = {}
        for g, v in d.items():
            if not v['pos'] or not v['neg']: continue
            ps, ng = np.maximum(np.array(v['pos']), 1e-3), np.maximum(np.array(v['neg']), 1e-3)
            (ms, ss), (mo, so) = lognorm(ps), lognorm(ng)
            llr = lambda x: np.clip(lpdf(x, ms, ss) - lpdf(x, mo, so), -np.log(8), np.log(8))
            o[g + ' s'] = {'n_same': len(ps), 'n_other': len(ng), 'quantiles_pct': qs,
                           'same': np.percentile(ps, qs).round(3).tolist(), 'other': np.percentile(ng, qs).round(3).tolist(),
                           'lognormal_same': [round(ms, 3), round(ss, 3)], 'lognormal_other': [round(mo, 3), round(so, 3)],
                           'other_pairs_lr_le_quarter': round(float((llr(ng) <= -np.log(4)).mean()), 3),
                           'same_pairs_lr_ge_4': round(float((llr(ps) >= np.log(4)).mean()), 3),
                           'same_pairs_lr_le_quarter': round(float((llr(ps) <= -np.log(4)).mean()), 3)}
        out.setdefault(rk, {})[KIND.get(name, name)] = o
    json.dump(out, open(dist, 'w'), indent=1)
    print(json.dumps(out, indent=1))
