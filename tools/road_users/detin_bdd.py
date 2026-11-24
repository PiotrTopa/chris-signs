#!/usr/bin/env python3
"""detin_bdd.py SAMPLES.json REPLAY.jsonl LIST DETIN_TEST: the low-score list and the flags against BDD100K labels.

SAMPLES.json   the FiftyOne export of BDD100K val (detections with label, bounding_box, occluded, truncated)
REPLAY.jsonl   lights_replay --no-track --road-users --ru-low X --ru-flags on the images of LIST (one line per image)
LIST           the image files, in the order given to lights_replay
DETIN_TEST     build/detin_test (its --flags mode runs signd's flag code on the label boxes)

Prints: (A) the flags computed on the LABEL boxes against the labels' own attributes (the rule by itself);
(B) the flags of the detector's boxes against the attributes of the label each box matches; (C) what the low
list adds: labels found by the first list, by the low list at several floors, and what the low boxes are.
"""
import json, os, subprocess, sys
import numpy as np

samples, replay, lst, tool = sys.argv[1:5]
RU = {'car', 'truck', 'bus', 'train', 'pedestrian', 'rider', 'motorcycle', 'bicycle', 'other vehicle', 'other person', 'trailer'}
W, H = 1280, 720
files = [os.path.basename(l.strip()) for l in open(lst) if l.strip()]
want = set(files)
gt = {}
for s in json.load(open(samples))['samples']:
    name = os.path.basename(s['filepath'])
    if name not in want: continue
    g = []
    for d in (s.get('detections') or {}).get('detections', []):
        if d['label'] not in RU: continue
        x, y, w, h = d['bounding_box']
        g.append((x * W, y * H, (x + w) * W, (y + h) * H, bool(d.get('occluded')), bool(d.get('truncated')), d['label']))
    gt[name] = g

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)

def pr(name, flag, truth):
    flag, truth = np.asarray(flag, bool), np.asarray(truth, bool)
    tp = (flag & truth).sum()
    print('  %-34s flagged %5d of %5d; of the flagged %4.1f %% carry the label attribute; of the attributed %4.1f %% are flagged'
          % (name, flag.sum(), len(flag), 100.0 * tp / max(1, flag.sum()), 100.0 * tp / max(1, truth.sum())))

# ---- A: the rule on the label boxes
inp = ''.join('%s %.2f %.2f %.2f %.2f\n' % (n, *g[:4]) for n in files for g in gt.get(n, []))
out = subprocess.run([tool, '--flags', str(W), str(H)], input=inp, capture_output=True, text=True, check=True).stdout.split('\n')
rows = [l.split() for l in out if l.strip()]
flat = [g for n in files for g in gt.get(n, [])]
assert len(rows) == len(flat)
tr = np.array([int(r[5]) for r in rows]); occ = np.array([float(r[6]) for r in rows])
g_occ = np.array([g[4] for g in flat]); g_tr = np.array([g[5] for g in flat])
hh = np.array([g[3] - g[1] for g in flat])
print('A. the flags computed on the label boxes (%d road users of %d images)' % (len(flat), len(files)))
pr('trunc (any frame edge)', tr > 0, g_tr)
pr('occ > 0', occ > 0, g_occ)
for lo, hi in ((0, 0.001), (0.001, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)):
    m = (occ >= lo) & (occ < hi)
    print('    occ in [%.2f, %.2f): %5d boxes, labelled occluded %4.1f %%' % (lo, hi, m.sum(), 100.0 * g_occ[m].mean() if m.any() else 0))
m = (occ == 0) & (tr == 0)
print('    neither flag: %d boxes, labelled occluded %.1f %% (things no box covers: poles, trees, kerbs, unlabelled vehicles), truncated %.1f %%'
      % (m.sum(), 100.0 * g_occ[m].mean(), 100.0 * g_tr[m].mean()))

# ---- B, C: the detector's boxes
msgs = [json.loads(l) for l in open(replay)]
assert len(msgs) == len(files), (len(msgs), len(files))
B = dict(tr=[], gtr=[], occ=[], gocc=[])
floors = (0.05, 0.10, 0.15, 0.20, 0.30)
C = {f: dict(found=0, found_occ=0, found_small=0, n=0, new=0, dup=0, none=0) for f in floors}
n_gt = n_gt_occ = n_gt_small = hi_found = hi_found_occ = hi_found_small = n_hi = hi_tp = 0
EDGE = {'first list': [], 'low list (floor 0.10)': []}   # per matched box: (edge errors l t r b) / box height, IoU, class right
CLSMAP = {'person': 'pedestrian', 'rider': 'rider', 'car': 'car', 'truck': 'truck', 'bus': 'bus', 'train': 'train', 'motor': 'motorcycle', 'bike': 'bicycle'}
def edge_row(u, gg):
    hgt = max(1.0, gg[3] - gg[1])
    return [(u['box'][i] - gg[i]) / hgt for i in range(4)] + [iou(u['box'], gg), float(CLSMAP.get(u['cls']) == gg[6]), hgt]
for name, m in zip(files, msgs):
    g = gt.get(name, [])
    hi_boxes = m.get('road_users') or []
    used = [-1] * len(g)   # label -> index of the first-list box that found it
    for k, u in sorted(enumerate(hi_boxes), key=lambda t: -t[1]['score']):
        best, bj = 0.5, -1
        for j, gg in enumerate(g):
            if used[j] >= 0: continue
            v = iou(u['box'], gg)
            if v >= best: best, bj = v, j
        n_hi += 1
        if bj >= 0:
            used[bj] = k; hi_tp += 1
            EDGE['first list'].append(edge_row(u, g[bj]))
            B['tr'].append(bool(u.get('trunc'))); B['gtr'].append(g[bj][5])
            B['occ'].append(u.get('occ', 0.0)); B['gocc'].append(g[bj][4])
    small = [(gg[3] - gg[1]) < 30 for gg in g]
    n_gt += len(g); n_gt_occ += sum(gg[4] for gg in g); n_gt_small += sum(small)
    hi_found += sum(u >= 0 for u in used); hi_found_occ += sum(u >= 0 and gg[4] for u, gg in zip(used, g))
    hi_found_small += sum(u >= 0 and s for u, s in zip(used, small))
    for f in floors:
        c = C[f]
        got = [u >= 0 for u in used]
        for u in sorted(m.get('road_users_low') or [], key=lambda t: -t['score']):
            if u['score'] <= f: continue
            c['n'] += 1
            best, bj = 0.5, -1
            for j, gg in enumerate(g):
                v = iou(u['box'], gg)
                if v >= best and not (got[j] and bj >= 0): best, bj = v, j
            if bj < 0: c['none'] += 1
            elif got[bj]: c['dup'] += 1
            else:
                got[bj] = True; c['new'] += 1
                if f == 0.10: EDGE['low list (floor 0.10)'].append(edge_row(u, g[bj]))
        c['found'] += sum(got); c['found_occ'] += sum(a and gg[4] for a, gg in zip(got, g)); c['found_small'] += sum(a and s for a, s in zip(got, small))
print('B. the flags of the detector\'s boxes (first list, %d boxes that match a label at IoU >= 0.5)' % len(B['tr']))
pr('trunc (any frame edge)', B['tr'], B['gtr'])
pr('occ > 0', np.array(B['occ']) > 0, B['gocc'])
print('C. the low list. Labels: %d road users (%d labelled occluded, %d under 30 px tall); first list: %d boxes, %.1f %% on a label'
      % (n_gt, n_gt_occ, n_gt_small, n_hi, 100.0 * hi_tp / max(1, n_hi)))
print('   first list alone finds %.1f %% of the labels (occluded %.1f %%, under 30 px %.1f %%)'
      % (100.0 * hi_found / n_gt, 100.0 * hi_found_occ / n_gt_occ, 100.0 * hi_found_small / n_gt_small))
for f in floors:
    c = C[f]
    print('   floor %.2f: + %5d low boxes (%.2f per image): %4.1f %% on a label the first list missed, %4.1f %% on one it found, %4.1f %% on none'
          ' -> labels found %.1f %% (occluded %.1f %%, under 30 px %.1f %%)'
          % (f, c['n'], c['n'] / len(files), 100.0 * c['new'] / max(1, c['n']), 100.0 * c['dup'] / max(1, c['n']), 100.0 * c['none'] / max(1, c['n']),
             100.0 * c['found'] / n_gt, 100.0 * c['found_occ'] / n_gt_occ, 100.0 * c['found_small'] / n_gt_small))
print('D. how good a box is (boxes on a label at IoU >= 0.5; edge error as a share of the label\'s height, robust sigma = 1.48 MAD)')
for name, rows in EDGE.items():
    a = np.array(rows)
    if not len(a): continue
    sg = [1.4826 * np.median(np.abs(a[:, i] - np.median(a[:, i]))) for i in range(4)]
    print('   %-22s n %5d: left %.3f top %.3f right %.3f bottom %.3f; median IoU %.2f; class right %.0f %%; label height median %.0f px'
          % (name, len(a), *sg, np.median(a[:, 4]), 100 * a[:, 5].mean(), np.median(a[:, 6])))
    for lo, hi in ((0, 30), (30, 80), (80, 1e9)):
        m = (a[:, 6] >= lo) & (a[:, 6] < hi)
        if m.sum() > 20:
            sg = [1.4826 * np.median(np.abs(a[m, i] - np.median(a[m, i]))) * np.median(a[m, 6]) for i in range(4)]
            print('      height %3g-%-4g px (n %5d): edge sigma in px: left %.1f top %.1f right %.1f bottom %.1f' % (lo, hi, m.sum(), *sg))
