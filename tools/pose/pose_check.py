#!/usr/bin/env python3
"""pose_check.py OUT.jsonl [--f F --cv CV --cam-h H]: consistency checks of the road users' `pose` on a ride without
labels (the output of lights_replay --road-users --ru-pose MODEL --ru-view --ru-flags [--ru-scale] --name-fps 20).

There is no truth on a ride. What can be held against the head's answers:
  lamps      in the dark, the lamp pair's colour (red = the rear faces the camera, white = the front) against the
             heading's hemisphere
  motion     objects = chains of boxes over the passes. A chain that crosses the picture sideways at a constant
             size is a vehicle seen side-on, pointing the way it moves; a chain that keeps its size for seconds
             in the middle of the picture is the vehicle ahead, seen from behind
  contact    on boxes with no cut and no covered edge: the near-face row gn against the box's bottom edge, the
             corners against the box's sides, the centre inside the box; with a camera (--f --cv --cam-h: focal
             length, horizon row, camera height) the range of gn against the range from the box height and the
             head's own height
  time       successive boxes of one chain: the change of the heading against the stated sigma, flips, view class
             changes, the jitter of the ground points relative to the box
None of this proves an accuracy: the lamp colour and the motion classes are themselves rules with errors, and
agreement with the box bottom shows only that the head does not contradict the detector where the detector is good."""
import json, math, sys
import numpy as np

args = sys.argv[1:]
opt = lambda k, d=None: float(args[args.index(k) + 1]) if k in args else d
F, CV, CAMH = opt('--f'), opt('--cv'), opt('--cam-h')
VEH = ('car', 'truck', 'bus', 'train')


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def adiff(a, b): return (a - b + 180.0) % 360.0 - 180.0


msgs = []
for line in open(args[0]):
    if not line.startswith('{') or not line.rstrip().endswith('}'): continue
    m = json.loads(line)
    if m.get('road_users'): msgs.append(m)
W, H = msgs[0]['frame'] if msgs else (1280, 720)
# chains: a box continues the chain whose last box it overlaps best (IoU >= 0.4), within 0.65 s
chains, open_ = [], []
nbox = npose = 0
for m in msgs:
    t = m['ts_mono_ns'] * 1e-9
    open_ = [c for c in open_ if t - c[-1][0] <= 0.65]
    used = set()
    for u in m['road_users']:
        if u['cls'] not in VEH: continue
        nbox += 1
        if 'pose' in u: npose += 1
        best, bi = None, 0.4
        for k, c in enumerate(open_):
            if k in used or c[-1][0] >= t: continue
            v = iou(u['box'], c[-1][1]['box'])
            if v > bi: best, bi = k, v
        if best is None:
            c = []; chains.append(c); open_.append(c); used.add(len(open_) - 1)
        else:
            c = open_[best]; used.add(best)
        c.append((t, u))
print('%d messages with road users, %d vehicle boxes, %d with a pose (%.0f %%), %d chains (%d of 5 boxes or more)' % (
    len(msgs), nbox, npose, 100.0 * npose / max(1, nbox), len(chains), sum(len(c) >= 5 for c in chains)))
allp = [(t, u) for c in chains for t, u in c if 'pose' in u]
if not allp: sys.exit(0)

def whole(u): return not u.get('clip') and not u.get('trunc') and not u.get('occ_edges') and u.get('occ', 0) < 0.05

P = lambda u: u['pose']
yaw = np.array([P(u)['yaw'] for _, u in allp]); ystd = np.array([P(u)['yaw_std'] for _, u in allp])
pm = np.array([P(u)['p'] for _, u in allp]); pf = np.array([P(u)['flip'] for _, u in allp])
vp = [P(u)['vp'] for _, u in allp]
print('\n-- what the head says (all poses)')
print('  view class: ' + ', '.join('%s %.0f %%' % (v, 100.0 * vp.count(v) / len(vp)) for v in ('rear', 'front', 'side', 'threequarter')))
print('  stated yaw sigma: median %.1f deg, 90 %% point %.1f; p_main median %.2f, under 0.7 in %.0f %%; p_flip median %.2f, over 0.2 in %.0f %%' % (
    np.median(ystd), np.quantile(ystd, 0.9), np.median(pm), 100 * (pm < 0.7).mean(), np.median(pf), 100 * (pf > 0.2).mean()))
kind = np.array([P(u)['kind'] for _, u in allp])
print('  kind: vehicle %.0f %%, person %.0f %%, two-wheeler %.0f %% (the detector called all of them vehicles)' % tuple(100 * (kind.argmax(1) == k).mean() for k in range(3)))

# ---- lamps
rows = [(P(u), u['lamps']['c']) for _, u in allp if 'lamps' in u]
print('\n-- heading against the lamp colour (dark frames): %d boxes with a lamp pair and a pose' % len(rows))
for col, want in (('red', 'rear half (|yaw| < 90)'), ('white', 'front half (|yaw| > 90)')):
    s = [p for p, c in rows if c == col]
    if not s: continue
    def good(p): return (abs(p['yaw']) < 90) == (col == 'red')
    g = np.array([good(p) for p in s])
    sure = np.array([p['p'] >= 0.8 and p['flip'] < 0.1 for p in s])
    vv = [p['vp'] for p in s]
    print('  %-5s n %4d: heading in the %s %.0f %%; of the %d the head is sure of (p >= 0.8, flip < 0.1): %.0f %%; view said: %s' % (
        col, len(s), want, 100 * g.mean(), sure.sum(), 100 * g[sure].mean() if sure.any() else float('nan'),
        ', '.join('%s %d' % (v, vv.count(v)) for v in ('rear', 'front', 'side', 'threequarter'))))
    # is the stated flip probability honest against the lamp colour?
    wrong_p = np.array([p['flip'] if not good(p) else 0 for p in s])
    print('        mean stated p_flip %.2f, mean stated p_main %.2f; share against the colour %.2f' % (np.mean([p['flip'] for p in s]), np.mean([p['p'] for p in s]), 1 - g.mean()))

# ---- motion
print('\n-- heading against the motion of chains')
# The camera's own turning moves every box sideways: the common sideways speed of the boxes (median over the chains
# seen in two successive messages, at least three of them) is taken out before a chain counts as crossing.
step = {}
for c in chains:
    for (t0, u0), (t1, u1) in zip(c[:-1], c[1:]):
        if t1 - t0 <= 0.45: step.setdefault(round(t1, 3), []).append(((u1['box'][0] + u1['box'][2]) - (u0['box'][0] + u0['box'][2])) / 2 / (t1 - t0))
common = {t: float(np.median(v)) for t, v in step.items() if len(v) >= 3}
cross, lead = [], []
for c in chains:
    c = [(t, u) for t, u in c]
    if len(c) < 5 or c[-1][0] - c[0][0] < 0.8: continue
    t = np.array([q[0] for q in c]); b = np.array([q[1]['box'] for q in c])
    cx = (b[:, 0] + b[:, 2]) / 2; w = b[:, 2] - b[:, 0]; h = b[:, 3] - b[:, 1]
    dur = t[-1] - t[0]
    g = [common[round(q, 3)] for q in t[1:] if round(q, 3) in common]
    vx_raw = np.polyfit(t, cx, 1)[0]
    turning = len(g) < 0.5 * (len(t) - 1)                # no other boxes to tell the camera's turning by
    vx = (vx_raw - (np.mean(g) if g else 0.0)) / np.median(w)          # box widths a second, against the other boxes
    gr = np.polyfit(t, np.log(h), 1)[0]                  # 1/s
    poses = [P(q[1]) for q in c if 'pose' in q[1] and not q[1].get('trunc')]
    if len(poses) < 3: continue
    if abs(vx) > 0.8 and abs(gr) < 0.2 and abs(cx[-1] - cx[0]) > 1.5 * np.median(w) and not turning and abs(vx_raw) > 0.8 * np.median(w):
        cross.append((vx, poses))
    elif dur >= 3.0 and abs(gr) < 0.08 and abs(vx) < 0.15 and abs(np.median(cx) - W / 2) < 0.15 * W:
        lead.append(poses)
n = sum(len(p) for _, p in cross)
if n:
    ok_side = sum(45 <= abs(p['yaw']) <= 135 for _, ps in cross for p in ps)
    ok_dir = sum((p['yaw'] > 0) == (vx > 0) for vx, ps in cross for p in ps)
    print('  crossing (moves sideways against the other boxes by more than 0.8 box widths a second at a constant size): %d chains, %d poses: side-on (45-135 deg) %.0f %%, pointing the way it moves %.0f %%' % (
        len(cross), n, 100.0 * ok_side / n, 100.0 * ok_dir / n))
else:
    print('  crossing: no such chain')
n = sum(len(p) for p in lead)
if n:
    a = np.array([p['yaw'] for ps in lead for p in ps])
    print('  ahead (keeps its size for 3 s or more in the middle of the picture): %d chains, %d poses: |yaw| < 30 deg %.0f %%, < 90 deg %.0f %%, median |yaw| %.1f deg, view rear %.0f %%' % (
        len(lead), n, 100 * (np.abs(a) < 30).mean(), 100 * (np.abs(a) < 90).mean(), np.median(np.abs(a)),
        100.0 * sum(p['vp'] == 'rear' for ps in lead for p in ps) / n))
else:
    print('  ahead: no such chain')

# ---- contact
rows = [u for _, u in allp if whole(u)]
print('\n-- ground points against the box, %d boxes with no cut and no covered edge' % len(rows))
if rows:
    b = np.array([u['box'] for u in rows]); bw = b[:, 2] - b[:, 0]; bh = b[:, 3] - b[:, 1]
    gn = np.array([P(u)['gn'] for u in rows]); gc = np.array([P(u)['gc'] for u in rows])
    gl = np.array([P(u)['gl'] for u in rows]); gr_ = np.array([P(u)['gr'] for u in rows])
    d = (gn[:, 1] - b[:, 3]) / bh
    print('  gn row - box bottom: median %+.3f box heights (%.1f px), spread (16-84 %%) %.3f; |.| > 0.1 box heights in %.1f %%; in units of the stated sigma: 68 %% point %.2f' % (
        np.median(d), np.median(gn[:, 1] - b[:, 3]), (np.quantile(d, 0.84) - np.quantile(d, 0.16)) / 2, 100 * (np.abs(d) > 0.1).mean(),
        np.quantile(np.abs(gn[:, 1] - b[:, 3]) / np.maximum(gn[:, 2], 1e-3), 0.683)))
    print('  gn said seen (>= 0.5): %.0f %%; gc above gn (the centre is farther): %.0f %%; gc inside the box: %.0f %%' % (
        100 * (gn[:, 3] >= 0.5).mean(), 100 * (gc[:, 1] <= gn[:, 1] + 0.5).mean(),
        100 * ((gc[:, 0] >= b[:, 0]) & (gc[:, 0] <= b[:, 2]) & (gc[:, 1] >= b[:, 1]) & (gc[:, 1] <= b[:, 3] + 0.05 * bh)).mean()))
    print('  gl x - box left: median %+.3f box widths, spread %.3f; gr x - box right: median %+.3f, spread %.3f' % (
        np.median((gl[:, 0] - b[:, 0]) / bw), np.std((gl[:, 0] - b[:, 0]) / bw), np.median((gr_[:, 0] - b[:, 2]) / bw), np.std((gr_[:, 0] - b[:, 2]) / bw)))
    low = np.maximum(gl[:, 1], gr_[:, 1])
    print('  the lower of gl, gr against the box bottom: median %+.3f box heights' % np.median((low - b[:, 3]) / bh))
    if F and CV is not None and CAMH:
        ok = (gn[:, 1] > CV + 3) & (b[:, 3] > CV + 3)
        r_gn = F * CAMH / (gn[ok, 1] - CV); r_bb = F * CAMH / (b[ok, 3] - CV); r_gc = F * CAMH / np.maximum(gc[ok, 1] - CV, 1)
        hh = np.array([P(u)['dim'][2] for u in rows])[ok]
        r_h = F * hh / bh[ok]
        cls = np.array([u['cls'] for u in rows])[ok]
        r_p = F * np.where(cls == 'car', 1.6, 3.2) / bh[ok]
        print('  flat road, camera %.2f m above it, f %.0f, horizon row %.0f (%d boxes below the horizon):' % (CAMH, F, CV, ok.sum()))
        for a, b_ in ((0, 15), (15, 30), (30, 60), (60, 999)):
            s = (r_bb >= a) & (r_bb < b_)
            if s.sum() < 10: continue
            q = lambda x: '%+.2f (spread %.2f)' % (np.median(np.log(x)), (np.quantile(np.log(x), 0.84) - np.quantile(np.log(x), 0.16)) / 2)
            print('    %3d-%3d m n %4d: ln(range gn / range box bottom) %s; ln(box height with the head\'s height / gn) %s; ln(box height with the class prior / gn) %s; centre - near %.1f m median' % (
                a, b_, s.sum(), q(r_gn[s] / r_bb[s]), q(r_h[s] / r_gn[s]), q(r_p[s] / r_gn[s]), np.median(r_gc[s] - r_gn[s])))

# ---- time
dy, pull, flips, vsw, dgn, dgn_pull, ddim, dts = [], [], 0, 0, [], [], [], []
for c in chains:
    q = [(t, u) for t, u in c if 'pose' in u]
    for (t0, u0), (t1, u1) in zip(q[:-1], q[1:]):
        if t1 - t0 > 0.45: continue
        p0, p1 = P(u0), P(u1)
        d = abs(adiff(p1['yaw'], p0['yaw']))
        dts.append(t1 - t0)
        if d > 120: flips += 1
        else:
            dy.append(d); pull.append(d / math.hypot(p0['yaw_std'], p1['yaw_std']))
        vsw += p0['vp'] != p1['vp']
        if whole(u0) and whole(u1):
            h0 = u0['box'][3] - u0['box'][1]; h1 = u1['box'][3] - u1['box'][1]
            a = (p0['gn'][1] - u0['box'][3]) / h0; b = (p1['gn'][1] - u1['box'][3]) / h1
            dgn.append(abs(a - b)); dgn_pull.append(abs(a - b) / math.hypot(p0['gn'][2] / h0, p1['gn'][2] / h1))
        ddim.append(abs(math.log(p1['dim'][0] / p0['dim'][0])))
n = len(dy) + flips
print('\n-- successive boxes of one chain (%d pairs, %.2f s apart in the median)' % (n, np.median(dts) if dts else 0))
if n:
    dy = np.array(dy); pull = np.array(pull)
    print('  heading: flips (change > 120 deg) %.1f %%; else median change %.1f deg, 90 %% point %.1f; in units of the two stated sigmas: 68 %% point %.2f (1 = honest if the object did not turn and the errors were independent), > 3 in %.1f %%' % (
        100.0 * flips / n, np.median(dy), np.quantile(dy, 0.9), np.quantile(pull, 0.683), 100 * (pull > 3).mean()))
    print('  view class changes in %.1f %% of the pairs; length (ln) changes by %.3f in the median' % (100.0 * vsw / n, np.median(ddim)))
if dgn:
    print('  gn row relative to the box (whole boxes, %d pairs): median change %.3f box heights, 90 %% point %.3f; in units of the stated sigmas: 68 %% point %.2f' % (
        len(dgn), np.median(dgn), np.quantile(dgn, 0.9), np.quantile(dgn_pull, 0.683)))
