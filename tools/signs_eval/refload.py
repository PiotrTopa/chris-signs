"""Reference detections: the whole-frame view and the six full-scale tiles of cache_dets.py merged by NMS."""
import numpy as np


def iou1(b, B):
    ix = np.maximum(0, np.minimum(b[2], B[:, 2]) - np.maximum(b[0], B[:, 0]))
    iy = np.maximum(0, np.minimum(b[3], B[:, 3]) - np.maximum(b[1], B[:, 1]))
    inter = ix * iy
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter + 1e-9)


def load_ref(base, names, min_score=0.35, extra=("T6",)):
    """-> dict frame -> list of (box, score, cls_name, cls_prob, src) after cross-view NMS."""
    zs = {"A": np.load(f"{base}.A.npz")}
    for e in extra:
        zs[e] = np.load(f"{base}.{e}.npz")
    frames = set(int(f) for f in zs[extra[0]]["frames"]) if extra else set(int(f) for f in zs["A"]["frames"])
    per = {}
    for src, z in zs.items():
        sel = z["score"] >= min_score
        for f, b, s, c, p in zip(z["frame"][sel], z["box"][sel], z["score"][sel], z["cls_raw"][sel], z["cls_prob"][sel]):
            if int(f) in frames:
                per.setdefault(int(f), []).append((b, float(s), names[int(c)], float(p), src))
    out = {}
    for f in sorted(frames):
        d = sorted(per.get(f, []), key=lambda q: -q[1])
        kept = []
        for q in d:
            if not kept or iou1(q[0], np.array([k[0] for k in kept])).max() <= 0.45:
                kept.append(q)
        out[f] = kept
    return out
