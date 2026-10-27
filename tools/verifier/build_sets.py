#!/usr/bin/env python3
"""Collect every labelled 32x32 sample (box region, RGB) with its source, split and label into one npz.
label 1 = real sign, 0 = lamp/light/other.
usage: build_sets.py HARVEST_DIR GTSRB_DATA_DIR OUT.npz
HARVEST_DIR holds krakow/ (harvest_ride.py + tracks.json from a linker + labels_false.json {track id: lamp|light|other},
unlabelled tracks with >= 2 round red votes count as real), bdd/ (harvest_bdd.py), bddblobs/ and bddlights/
(mine_bdd_blobs.py, the second with "lights"); any of them may be missing."""
import sys, os, json, csv, hashlib, numpy as np, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glow32, pipe
from PIL import Image
H, G, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
ex = lambda p: os.path.exists(H + '/' + p)
X, Y, SRC, SPL, GL, META = [], [], [], [], [], []
def add(im_box, y, src, split, meta=''):
    a = glow32.arr32(im_box); X.append(a.astype(np.uint8)); Y.append(y); SRC.append(src); SPL.append(split); GL.append(glow32.glow(a)); META.append(meta)
# Krakow harvest (real RR detections of tracks with >= 2 red votes; false = labelled tracks)
dets = {(x['f'], x['k']): x for x in map(json.loads, open(H + '/krakow/dets.jsonl'))} if ex('krakow/tracks.json') else {}
lab = json.load(open(H + '/krakow/labels_false.json')) if ex('krakow/labels_false.json') else {}
for t in (json.load(open(H + '/krakow/tracks.json')) if dets else []):
    rd = [dets[tuple(fk)] for fk in t['dets'] if dets[tuple(fk)]['red']]
    if len(rd) < 2: continue
    L = lab.get(str(t['tid']), 'real')
    for x in rd:
        im = Image.open(f"{H}/krakow/crops/{x['f']:05d}_{x['k']}.png")
        add(glow32.box_from_margin_crop(im), int(L == 'real'), 'krakow_' + L, 'train' if x['f'] < 6000 else 'test', f"{t['tid']}:{x['f']}")
# BDD detector harvest
for r in (map(json.loads, open(H + '/bdd/dets.jsonl')) if ex('bdd/dets.jsonl') else []):
    g = r['gt']; inA = lambda k: g.get(k, (0,))[0]
    if inA('traffic sign') >= 0.5: y, s = 1, 'bdd_sign'
    elif inA('traffic light') >= 0.5: y, s = 0, 'bdd_det_light'
    elif max(inA('car'), inA('truck'), inA('bus')) >= 0.8: y, s = 0, 'bdd_det_vehicle'
    else: continue
    add(glow32.box_from_margin_crop(Image.open(H + '/bdd/' + r['crop'])), y, s + ('_night' if r.get('tod') == 'night' else ''), r['split'], r['img'])
# BDD blobs (subsample per split)
random.seed(0)
bl = [dict(r, d=dd) for dd in ('bddblobs', 'bddlights') if ex(dd + '/blobs.json') for r in json.load(open(H + '/' + dd + '/blobs.json'))]
random.shuffle(bl)
cnt = {}
for r in bl:
    k = (r['split'], r['src'] == 'traffic light')
    if cnt.get(k, 0) >= (3000 if r['split'] == 'train' else 1000) // (2 if k[1] else 1): continue
    cnt[k] = cnt.get(k, 0) + 1
    add(glow32.box_from_margin_crop(Image.open(H + '/' + r['d'] + '/' + r['crop'])), 0, 'bdd_blob_' + ('light' if r['src'] == 'traffic light' else 'vehicle') + ('_night' if r.get('tod') == 'night' else ''), r['split'], r['img'])
# GTSRB round red classes: 0-5,7,8 speed limits, 9,10 no overtaking, 14 stop, 15 no vehicles, 16 no trucks, 17 no entry
RR = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 14, 15, 16, 17}
for split, fn, cap in (('train', 'Train.csv', 6000), ('test', 'Test.csv', 3000)):
    rows = [r for r in csv.DictReader(open(f'{G}/{fn}')) if int(r['ClassId']) in RR]
    random.shuffle(rows)
    for r in rows[:cap]:
        im = Image.open(G + '/' + r['Path'].replace('/code/', '')).convert('RGB')
        add(im.crop((int(r['Roi.X1']), int(r['Roi.Y1']), int(r['Roi.X2']), int(r['Roi.Y2']))), 1, 'gtsrb_' + r['ClassId'], split, r['Path'])
np.savez_compressed(OUT, X=np.stack(X), Y=np.array(Y), SRC=np.array(SRC), SPL=np.array(SPL), GL=np.array(GL), META=np.array(META))
import collections
print(collections.Counter(zip(SRC, SPL)))
