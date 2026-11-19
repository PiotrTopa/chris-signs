#!/usr/bin/env python3
"""score.py LABEL_DIR [--thr person=0.4,...]: the detector's class against the labels of sheets.py dets
(labels written before scoring). Per-class confusion (rows: label, columns: the detector's class after rider
pairing; a box below its class threshold counts as `dropped`), precision per detector class, and the split by
the sample's sessions. `van` is labelled where it is clearly a van; the detector has no van class, so `car` is
the right answer for a van and is counted as such (and listed)."""
import argparse, collections, json, os

NAMES = ["person", "rider", "car", "truck", "bus", "train", "motor", "bike"]
THR = {"person": 0.40, "rider": 0.35, "car": 0.40, "truck": 0.40, "bus": 0.40, "train": 0.50, "motor": 0.35, "bike": 0.35}
TO_LABEL = {"person": "person", "rider": "bike", "bike": "bike", "motor": "motorbike", "car": "car", "truck": "truck",
            "bus": "bus", "train": "tram"}
OK = {("van", "car"), ("van", "van"), ("car", "generic"), ("van", "generic")}   # generic = the car shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--thr", default="")
    ap.add_argument("--screen", action="store_true",
                    help="what the car's screen would draw from a single detection: a vehicle class below 0.5 is the "
                         "generic car shape (`generic`), a person / bike / motorbike from 0.35")
    a = ap.parse_args()
    thr = dict(THR)
    for kv in filter(None, a.thr.split(",")):
        k, v = kv.split("=")
        thr[k] = float(v)
    sample = json.load(open(os.path.join(a.dir, "sample.json")))
    labels = {int(l.split()[0]): l.split()[1] for l in open(os.path.join(a.dir, "labels.txt")) if l.strip()}
    conf = collections.Counter()
    for n, d in enumerate(sample):
        if n not in labels:
            continue
        c = d["cls"]
        pred = TO_LABEL[c] if d["score"] > thr[c] else "dropped"
        if d.get("rider") and c in ("bike", "motor"):
            pred = TO_LABEL[c]
        if a.screen and pred in ("truck", "bus", "tram") and d["score"] < 0.5:
            pred = "generic"
        conf[(labels[n], pred)] += 1
    rows = sorted({k[0] for k in conf})
    cols = sorted({k[1] for k in conf})
    print("label \\ detector".ljust(16) + "".join(c[:9].rjust(10) for c in cols) + "   n")
    for r in rows:
        print(r.ljust(16) + "".join(str(conf[(r, c)] or ".").rjust(10) for c in cols) + "   %d" % sum(conf[(r, c)] for c in cols))
    print()
    for c in cols:
        if c == "dropped":
            continue
        n = sum(conf[(r, c)] for r in rows)
        good = sum(conf[(r, c)] for r in rows if r == c or (r, c) in OK)
        print("%-10s precision %3d / %3d = %.2f" % (c, good, n, good / max(1, n)))
    allp = sum(v for (r, c), v in conf.items() if c != "dropped")
    good = sum(v for (r, c), v in conf.items() if c != "dropped" and (r == c or (r, c) in OK))
    print("all kept    %d / %d = %.2f ; dropped by thresholds %d" % (good, allp, good / max(1, allp),
                                                                     sum(v for (r, c), v in conf.items() if c == "dropped")))


if __name__ == "__main__":
    main()
