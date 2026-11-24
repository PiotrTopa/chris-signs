#!/usr/bin/env python3
"""low_check.py OUT.jsonl [FLOOR...]: what the low-score list (signd --lights-ru-low) gives a tracker on a ride,
without labels. Input: lights_replay --road-users --ru-low X (one line per pass).

Objects: chains of the first list's boxes (a box continues the chain whose last box, at most 1.2 s old, overlaps
it by IoU >= 0.25; same class group). A GAP is a pass between two boxes of one chain in which the first list has
no box of it although the view of that pass covers where the object must be (the box interpolated between its
neighbours lies inside the pass's region). Printed per floor: the share of the gap passes in which a low box
lies on the interpolated box (IoU >= 0.3): what a second association stage can pick up; and what the other low
boxes are: on a first-list box's chain position (duplicates), near the END of a chain (within 1.2 s after its last
box, IoU >= 0.3 with its extrapolation: the object fading out), or on nothing (clutter a tracker must not start
objects from)."""
import json, sys
out = sys.argv[1]
floors = [float(a) for a in sys.argv[2:]] or [0.10, 0.15, 0.20, 0.30]
G = {'person': 0, 'rider': 0, 'motor': 0, 'bike': 0}

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)

msgs = [json.loads(l) for l in open(out)]
msgs = [m for m in msgs if m.get('view', 0) >= 0 and 'road_users' in m]
chains = []
for k, m in enumerate(msgs):
    t = m['ts_mono_ns'] * 1e-9
    used = set()
    for u in m['road_users'] or []:
        g = G.get(u['cls'], 1)
        best, bs = None, 0.25
        for ci, c in enumerate(chains):
            if ci in used or c['g'] != g or t - c['e'][-1][0] > 1.2 or c['e'][-1][1] == k: continue
            s = iou(u['box'], c['e'][-1][2])
            if s >= bs: best, bs = ci, s
        if best is None:
            chains.append({'g': g, 'e': [(t, k, u['box'])]})
        else:
            chains[best]['e'].append((t, k, u['box'])); used.add(best)
# gaps: per pass index, the interpolated boxes of chains that have no box there
gaps = {}   # pass -> [box]
ends = {}   # pass -> [box] (extrapolated, held) for 1.2 s after a chain's last box
for c in chains:
    e = c['e']
    for (t0, k0, b0), (t1, k1, b1) in zip(e, e[1:]):
        for k in range(k0 + 1, k1):
            a = (msgs[k]['ts_mono_ns'] * 1e-9 - t0) / (t1 - t0)
            gaps.setdefault(k, []).append((c['g'], [b0[i] + a * (b1[i] - b0[i]) for i in range(4)]))
    if len(e) >= 2:
        tl, kl, bl = e[-1]
        for k in range(kl + 1, len(msgs)):
            if msgs[k]['ts_mono_ns'] * 1e-9 - tl > 1.2: break
            ends.setdefault(k, []).append((c['g'], bl))
def inside(b, roi):
    return b[0] >= roi[0] and b[2] <= roi[2] and b[1] >= roi[1] and (b[1] + b[3]) / 2 <= roi[3]
n_pass = len(msgs); n_hi = sum(len(m['road_users'] or []) for m in msgs)
n_gap = sum(1 for k, v in gaps.items() for g, b in v if inside(b, msgs[k]['roi']))
long_chains = sum(1 for c in chains if len(c['e']) >= 3)
print('%s: %d passes, first list %d boxes (%.2f a pass), %d chains (%d of 3 or more boxes); gap passes inside the view: %d (%.0f %% of the chains\' passes)'
      % (out, n_pass, n_hi, n_hi / max(1, n_pass), len(chains), long_chains, n_gap, 100.0 * n_gap / max(1, n_gap + sum(len(c['e']) for c in chains if len(c['e']) >= 2))))
for f in floors:
    filled = n_low = on_gap = on_end = on_hi = 0
    for k, m in enumerate(msgs):
        low = [u for u in (m.get('road_users_low') or []) if u['score'] > f]
        n_low += len(low)
        gl = [(g, b) for g, b in gaps.get(k, []) if inside(b, m['roi'])]
        got = [False] * len(gl)
        for u in low:
            g = G.get(u['cls'], 1)
            hit = False
            for i, (gg, b) in enumerate(gl):
                if gg == g and iou(u['box'], b) >= 0.3:
                    if not got[i]: got[i] = True
                    hit = True
            if hit: on_gap += 1; continue
            if any(gg == g and iou(u['box'], b) >= 0.3 for gg, b in ends.get(k, [])): on_end += 1; continue
            if any(iou(u['box'], h['box']) >= 0.3 for h in m['road_users'] or []): on_hi += 1
        filled += sum(got)
    print('  floor %.2f: %5d low boxes (%.2f a pass): gap passes filled %d of %d (%.0f %%); of the low boxes %.0f %% on a gap, %.0f %% after a chain\'s end, %.0f %% beside a first-list box, %.0f %% on nothing known'
          % (f, n_low, n_low / max(1, n_pass), filled, n_gap, 100.0 * filled / max(1, n_gap), 100.0 * on_gap / max(1, n_low), 100.0 * on_end / max(1, n_low),
             100.0 * on_hi / max(1, n_low), 100.0 * (n_low - on_gap - on_end - on_hi) / max(1, n_low)))
