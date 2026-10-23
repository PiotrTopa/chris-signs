#!/usr/bin/env python3
"""Compare the raw detections of two signd_replay runs (same inputs, e.g. --backend ort vs ncnn).

usage: compare_replay.py REF.jsonl TEST.jsonl [--iou 0.5] [--thr 0.35]

Both files come from `signd_replay --raw ...` (one `signs` line per frame, field "raw" = all
detections of that frame after NMS, before tracking: score, box in frame px, cls, cls_prob).
Per frame, detections are matched greedily by IoU (>= --iou); reported over the run:
  frames, ref/test detection counts, matched, missed (ref only), extra (test only)
  box error (px) mean / max over matches, |dscore| mean / max, classifier top-1 agreement over
  matches, |dcls_prob| mean / max, and the list of unmatched detections with their scores
  (a miss/extra with a score near the threshold is a borderline case, not a wrong model).
"""
import json
import sys


def load(p):
    return [m for m in map(json.loads, open(p)) if m.get("type") == "signs" and "raw" in m]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    thr_iou = float(sys.argv[sys.argv.index("--iou") + 1]) if "--iou" in sys.argv else 0.5
    ref, tst = load(a[0]), load(a[1])
    if len(ref) != len(tst):
        print(f"frame count differs: {len(ref)} vs {len(tst)}")
        sys.exit(1)
    n_ref = n_tst = n_match = cls_same = 0
    box_err, dscore, dprob, missed, extra = [], [], [], [], []
    for r, t in zip(ref, tst):
        R, T = list(r["raw"]), list(t["raw"])
        n_ref += len(R)
        n_tst += len(T)
        used = set()
        for d in R:
            best, bj = 0.0, -1
            for j, e in enumerate(T):
                if j in used:
                    continue
                v = iou(d["box"], e["box"])
                if v > best:
                    best, bj = v, j
            if bj >= 0 and best >= thr_iou:
                used.add(bj)
                e = T[bj]
                n_match += 1
                box_err.append(max(abs(p - q) for p, q in zip(d["box"], e["box"])))
                dscore.append(abs(d["score"] - e["score"]))
                dprob.append(abs(d["cls_prob"] - e["cls_prob"]))
                cls_same += int(d["cls"] == e["cls"])
            else:
                missed.append((r["seq"], d["score"], d["cls"]))
        for j, e in enumerate(T):
            if j not in used:
                extra.append((t["seq"], e["score"], e["cls"]))
    mean = lambda v: sum(v) / len(v) if v else 0.0
    mx = lambda v: max(v) if v else 0.0
    print(f"frames {len(ref)}: ref dets {n_ref}, test dets {n_tst}, matched {n_match} (IoU >= {thr_iou}), "
          f"missed {len(missed)}, extra {len(extra)}")
    print(f"matches: box err px mean {mean(box_err):.2f} max {mx(box_err):.2f}; |dscore| mean {mean(dscore):.4f} "
          f"max {mx(dscore):.4f}; cls top-1 same {cls_same}/{n_match}; |dcls_prob| mean {mean(dprob):.4f} max {mx(dprob):.4f}")
    if missed:
        print("missed (seq, ref score, cls):", ", ".join(f"({s}, {sc:.3f}, {c})" for s, sc, c in missed))
    if extra:
        print("extra  (seq, test score, cls):", ", ".join(f"({s}, {sc:.3f}, {c})" for s, sc, c in extra))


if __name__ == "__main__":
    main()
