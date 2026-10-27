#!/usr/bin/env python3
"""Score sign classifiers (ONNX, 64x64, softmax output) on held-out crops: MTSD val (extract_crops.py shards) and the
BDD100K test split (extract_bdd.py), under the phone-like renderings of cls_aug.eval_sample.

usage: eval_cls.py --val DIR --bdd BDD.npz --model NAME=MODEL.onnx:CLASSES.txt [--model ...] [--out OUT.json]

Reported per model and condition (full resolution, phone size, phone size capped at 24 / 16 px):
  pos_top1       top-1 on MTSD val signs of the 400 classes (gt boxes, not ambiguous, phone size >= 12 px), merged names
                 (--gN stripped); also with detector boxes (detpos)
  pos_macro      mean per-class recall over classes with >= 5 val samples
  other_claim    share of `other-sign` val signs given one of the 400 classes with p >= 0.6 (a vote in signd)
  neg_claim      share of not-a-sign crops given a sign class (any of the 400 + other-sign) with p >= 0.6:
                 MTSD detector false positives (detneg), random boxes (rand), BDD lamps / red lights / any lights /
                 detector false positives (test split)
  confusions     val signs of a family voted (p >= 0.6) as a known wrong class: warning triangles -> priority-road,
                 go-straight / C-5 discs -> one-way-straight, keep-right -> no-entry, other blue round -> no-entry
"""
import argparse, glob, json, os, re, sys, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cls_aug  # noqa: E402

GATE = 0.6
merge = lambda n: re.sub(r'--g\d+$', '', n)


def run_model(path, ims):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    s = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    out = []
    for im in ims:
        x = (im.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        p = s.run(None, {'images': np.ascontiguousarray(x)})[0][0]
        k = int(p.argmax()); out.append((k, float(p[k])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', required=True); ap.add_argument('--bdd'); ap.add_argument('--model', action='append', required=True)
    ap.add_argument('--out'); ap.add_argument('--max-neg', type=int, default=4000); ap.add_argument('--max-other', type=int, default=4000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    S = []   # (x, meta, group)
    for f in sorted(glob.glob(f'{a.val}/*.npz')):
        d = np.load(f); M = json.loads(str(d['meta'])); X = d['X']
        for i, m in enumerate(M):
            fl = m.get('flags', [])
            if 'ambiguous' in fl or 'dummy' in fl: continue
            if m['kind'] in ('gt', 'detpos'):
                if cls_aug.phone_size(m) < 12: continue
                g = ('other' if m['label'] == 'other-sign' else 'pos') + ('_det' if m['kind'] == 'detpos' else '')
            elif m['kind'] in ('detneg', 'rand'):
                g = 'neg_mtsd_' + m['kind']
            else: continue
            S.append((X[i], m, g))
    if a.bdd:
        d = np.load(a.bdd); M = json.loads(str(d['meta'])); X = d['X']
        for i, m in enumerate(M):
            if m.get('split') == 'test': S.append((X[i], m, 'neg_bdd_' + m['kind'] + ('_night' if m.get('tod') == 'night' else '')))
    by = collections.defaultdict(list)
    for t in S: by[t[2]].append(t)
    for g in list(by):
        cap = a.max_other if g.startswith('other') else (a.max_neg if g.startswith('neg') else None)
        if cap and len(by[g]) > cap:
            sel = rng.choice(len(by[g]), cap, replace=False); by[g] = [by[g][i] for i in sorted(sel)]
    print({g: len(v) for g, v in sorted(by.items())}, flush=True)
    conds = ['full', 'phone', 24, 16]
    models = []
    for spec in a.model:
        name, rest = spec.split('=', 1); path, cls = rest.rsplit(':', 1)
        models.append((name, path, [l.strip() for l in open(cls) if l.strip()]))
    res = {}
    fam = {
        'warning_to_priority_road': (lambda n: n.startswith('warning--'), lambda n: merge(n) == 'regulatory--priority-road'),
        'go_straight_to_one_way': (lambda n: re.match(r'regulatory--go-straight(--g\d+)?$', n) is not None, lambda n: merge(n).startswith('regulatory--one-way')),
        'keep_right_to_no_entry': (lambda n: merge(n) in ('regulatory--keep-right', 'regulatory--keep-left'), lambda n: merge(n) == 'regulatory--no-entry'),
        'blue_mandatory_to_round_red': (lambda n: re.match(r'regulatory--(go-straight|turn-|keep-|pass-on-either|roundabout|go-straight-or-turn)', n) is not None,
                                        lambda n: merge(n).startswith(('regulatory--no-', 'regulatory--maximum-speed', 'regulatory--height', 'regulatory--weight'))),
    }
    for name, path, names in models:
        R = {}
        for c in conds:
            row = {}
            for g, items in sorted(by.items()):
                ims = [cls_aug.eval_sample(x, m, c) for x, m, _ in items]
                pr = run_model(path, ims)
                if g.startswith('pos'):
                    ok = [merge(names[k]) == merge(m['label']) for (k, p), (x, m, _) in zip(pr, items)]
                    row[g + '_top1'] = round(float(np.mean(ok)), 4)
                    pc = collections.defaultdict(list)
                    for o, (x, m, _) in zip(ok, items): pc[merge(m['label'])].append(o)
                    row[g + '_macro'] = round(float(np.mean([np.mean(v) for v in pc.values() if len(v) >= 5])), 4)
                    row[g + '_n'] = len(items)
                    for fk, (src, dst) in fam.items():
                        sel = [(k, p) for (k, p), (x, m, _) in zip(pr, items) if src(m['label'])]
                        bad = sum(1 for k, p in sel if p >= GATE and dst(names[k]) and not src(names[k]))
                        row[f'{g}_{fk}'] = f'{bad}/{len(sel)}'
                    if g == 'pos' and c == 'phone':
                        R['per_class_phone'] = {k: [round(float(np.mean(v)), 3), len(v)] for k, v in sorted(pc.items())}
                elif g.startswith('other'):
                    row[g + '_claim'] = round(float(np.mean([p >= GATE and names[k] not in ('other-sign', 'background') for k, p in pr])), 4)
                    row[g + '_n'] = len(items)
                else:
                    row[g + '_claim'] = round(float(np.mean([p >= GATE and names[k] != 'background' for k, p in pr])), 4)
                    row[g + '_n'] = len(items)
            R[str(c)] = row
            print(name, c, json.dumps(row), flush=True)
        res[name] = R
    if a.out: json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
