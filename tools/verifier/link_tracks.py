#!/usr/bin/env python3
"""Link harvested detections (dets.jsonl, every frame) into simple tracks (IoU >= 0.2 with the last box within 3 frames).
Writes tracks.json: list of {tid, f0, f1, n, dets:[(f,k)], cls_votes, red_frac}. usage: link_tracks.py HARVEST_DIR"""
import sys, json, collections
d = sys.argv[1]
dets = [json.loads(l) for l in open(d + '/dets.jsonl')]
byf = collections.defaultdict(list)
for x in dets: byf[x['f']].append(x)
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-6)
live, done, nid = [], [], 0
for f in sorted(byf):
    live2 = [t for t in live if f - t['last'] <= 3]
    done += [t for t in live if f - t['last'] > 3]; live = live2
    used = set()
    for x in sorted(byf[f], key=lambda x: -x['score']):
        best, bv = None, 0.2
        for t in live:
            if id(t) in used: continue
            v = iou(t['box'], x['box'])
            if v > bv: best, bv = t, v
        if best is None:
            nid += 1; best = dict(tid=nid, f0=f, dets=[], votes=collections.Counter()); live.append(best)
        used.add(id(best)); best['box'] = x['box']; best['last'] = f; best['dets'].append([f, x['k']])
        best['votes'][x['cls']] += 1
done += live
out = []
for t in done:
    n = len(t['dets'])
    out.append(dict(tid=t['tid'], f0=t['f0'], f1=t['last'], n=n, dets=t['dets'], cls=t['votes'].most_common(1)[0][0],
                    votes=dict(t['votes'])))
json.dump(out, open(d + '/tracks.json', 'w'))
import re
red = [t for t in out if any(re.match(r'^regulatory--(no-|maximum-speed-limit|road-closed|height-limit|weight-limit|width-limit|stop)', c) for c in t['votes'])]
print(len(out), 'tracks;', len(red), 'with any red-class vote;', sum(1 for t in red if t['n'] >= 3), 'of them n>=3')
