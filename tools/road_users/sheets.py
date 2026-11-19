#!/usr/bin/env python3
"""sheets.py: contact sheets for labelling road users by eye, BEFORE anything is scored (roadsense lane L42).

  sheets.py dets SCAN.npz OUT_DIR [--per-class N] [--min-h PX] [--seed S]
      a stratified sample of the detector's boxes (views 0 and 1 of the phone's unit, riders paired as signd
      does, score >= 0.3, box height >= --min-h), drawn WITHOUT the detector's class: crop with context, the box
      as a thin grey frame, a number. Writes OUT_DIR/sheet_NN.png and OUT_DIR/sample.json (box, frame, view, the
      detector's class kept out of the picture). Labels go into OUT_DIR/labels.txt, one line per number:
      `<n> <label>` with label in person bike motorbike car van truck bus tram other.
  sheets.py lead SESSION_DIR... --out OUT_DIR [--n N]
      frames where the road model has a lead (prob >= 0.5, 3..70 m): the lead's projected rear (2 m x 2 m) drawn
      on a crop; OUT_DIR/lead_sample.json, labels `<n> <label>` (label `none` = no vehicle there / unreadable).
"""
import argparse, json, os, random, sys
import numpy as np, cv2

NAMES = ["person", "rider", "car", "truck", "bus", "train", "motor", "bike"]
CELL_W, CELL_H, COLS, ROWS = 200, 150, 6, 5


def pair_riders(dets):
    """dets: list of dict(cls, box, score). Same rule as signd pair_riders()."""
    gone, taken = set(), set()
    for i, r in enumerate(dets):
        if r["cls"] != "rider":
            continue
        best, best_ov = None, 0.0
        for j, b in enumerate(dets):
            if b["cls"] not in ("motor", "bike") or j in taken:
                continue
            rb, bb = r["box"], b["box"]
            ox = min(rb[2], bb[2]) - max(rb[0], bb[0])
            ov = ox / max(1.0, min(rb[2] - rb[0], bb[2] - bb[0]))
            if ov < 0.5 or rb[3] < bb[1] or (rb[1] + rb[3]) > (bb[1] + bb[3]) or ov <= best_ov:
                continue
            best, best_ov = j, ov
        if best is None:
            continue
        b = dets[best]
        b["box"] = [min(b["box"][0], r["box"][0]), min(b["box"][1], r["box"][1]), max(b["box"][2], r["box"][2]), max(b["box"][3], r["box"][3])]
        b["score"] = max(b["score"], r["score"])
        b["rider"] = True
        taken.add(best); gone.add(i)
    return [d for i, d in enumerate(dets) if i not in gone]


def load_dets(npz, views=(0, 1), thr=0.15):
    """{(frame index, view): [dict(cls, score, box, p, clip)]} with riders paired."""
    d = np.load(npz)
    rows, sc = d["rows"], d["scores"]
    out = {}
    for r, s in zip(rows, sc):
        fi, v = int(r[0]), int(r[1])
        if v not in views or r[2] < thr:
            continue
        roi = r[7:11]
        b = [float(x) for x in r[3:7]]
        clip = ("l" if b[0] <= roi[0] + 2 else "") + ("t" if b[1] <= roi[1] + 2 else "") + \
               ("r" if b[2] >= min(roi[2], 1280) - 2 else "") + ("b" if b[3] >= min(roi[3], 720) - 2 else "")
        out.setdefault((fi, v), []).append({"cls": NAMES[int(np.argmax(s))], "score": float(r[2]), "box": b,
                                            "p": [float(x) for x in s], "clip": clip, "rider": False})
    for k in out:
        out[k] = pair_riders(out[k])
    return out, list(d["frames"])


def cell(img, box, n, ctx=1.8, colour=(160, 160, 160)):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = max(x1 - x0, (y1 - y0) * 4 / 3, 48) * ctx
    h = w * CELL_H / CELL_W
    a0, b0 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
    a1, b1 = int(min(W, cx + w / 2)), int(min(H, cy + h / 2))
    crop = img[b0:b1, a0:a1]
    k = min(CELL_W / max(1, crop.shape[1]), (CELL_H - 14) / max(1, crop.shape[0]))
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * k)), max(1, int(crop.shape[0] * k))), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((CELL_H, CELL_W, 3), np.uint8)
    out[14:14 + crop.shape[0], :crop.shape[1]] = crop
    p0 = (int((x0 - a0) * k), int((y0 - b0) * k) + 14)
    p1 = (int((x1 - a0) * k), int((y1 - b0) * k) + 14)
    cv2.rectangle(out, p0, p1, colour, 1)
    cv2.putText(out, str(n), (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return out


def write_sheets(cells, out_dir, prefix="sheet"):
    per = COLS * ROWS
    paths = []
    for s in range(0, len(cells), per):
        sheet = np.zeros((ROWS * CELL_H, COLS * CELL_W, 3), np.uint8)
        for i, c in enumerate(cells[s:s + per]):
            r, q = divmod(i, COLS)
            sheet[r * CELL_H:(r + 1) * CELL_H, q * CELL_W:(q + 1) * CELL_W] = c
            cv2.rectangle(sheet, (q * CELL_W, r * CELL_H), ((q + 1) * CELL_W - 1, (r + 1) * CELL_H - 1), (60, 60, 60), 1)
        p = os.path.join(out_dir, "%s_%02d.png" % (prefix, s // per))
        cv2.imwrite(p, sheet)
        paths.append(p)
    return paths


def cmd_dets(a):
    dets, frames = load_dets(a.scan)
    pool = {}
    for (fi, v), lst in dets.items():
        for d in lst:
            if d["score"] < 0.3 or d["box"][3] - d["box"][1] < a.min_h:
                continue
            key = {"rider": "bike", "motor": "motor"}.get(d["cls"], d["cls"])
            pool.setdefault(key, []).append(dict(d, frame=frames[fi], fi=fi, view=v))
    rng = random.Random(a.seed)
    sample = []
    for k in sorted(pool):
        lst = pool[k]
        rng.shuffle(lst)
        sample += lst[:a.per_class]
        print(k, "pool", len(lst), "sampled", min(len(lst), a.per_class), file=sys.stderr)
    rng.shuffle(sample)                  # classes mixed on the sheets: the order says nothing
    os.makedirs(a.out, exist_ok=True)
    cells, cache = [], {}
    for n, d in enumerate(sample):
        img = cache.get(d["frame"])
        if img is None:
            cache.clear()
            img = cache[d["frame"]] = cv2.imread(d["frame"])
        cells.append(cell(img, d["box"], n))
    write_sheets(cells, a.out)
    with open(os.path.join(a.out, "sample.json"), "w") as f:
        json.dump(sample, f)
    print("sample", len(sample), file=sys.stderr)


def lead_box(x, y_right, f=975.0, cx=640.0, cy=360.0, h_cam=1.22, w=2.0, h=2.0):
    u0, u1 = cx + f * (y_right - w / 2) / x, cx + f * (y_right + w / 2) / x
    vb, vt = cy + f * h_cam / x, cy + f * (h_cam - h) / x
    return [u0, vt, u1, vb]


def cmd_lead(a):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "roadsense-l42", "chris-integrator"))
    from chris_integrator.replay import read_lines
    cands = []
    for s in a.sessions:
        have = {int(os.path.basename(p)[:-4]) for p in os.listdir(os.path.join(s, "frames")) if p.endswith(".jpg")}
        calib = None
        for m in read_lines(s):
            if m.get("type") == "calib" and calib is None:
                calib = m
            if m.get("type") != "road" or m.get("seq") not in have:
                continue
            ld = (m.get("leads") or [None])[0]
            if not ld or (ld.get("prob") or 0) < 0.5:
                continue
            x, y = ld["x"][0], ld["y"][0]
            if not 3.0 <= x <= 70.0:
                continue
            # stored as the road model gives it (projected as y right: on these supercombo recordings with the
            # car's calibration error, roadsense L32, the box sits about a car width right of the vehicle; the
            # labeller labels the vehicle ahead in the lane the box is nearest to)
            u = 640.0 + 975.0 * y / x
            if not 40.0 <= u <= 1240.0:
                continue
            c = calib or {}
            cands.append({"frame": os.path.join(s, "frames", "%d.jpg" % m["seq"]), "seq": m["seq"], "ts": m["ts_mono_ns"],
                          "session": os.path.basename(s.rstrip("/")), "x": x, "y": y, "prob": ld["prob"],
                          "box": lead_box(x, y, c.get("focal_px") or 975.0, c.get("cx") or 640.0, c.get("cy") or 360.0,
                                          c.get("height_m") or 1.22)})
    rng = random.Random(a.seed)
    rng.shuffle(cands)
    # spread over the drive: at most one frame per 20 s of a session
    picked, last = [], {}
    for c in sorted(cands, key=lambda c: (c["session"], c["ts"])):
        if c["ts"] - last.get(c["session"], -1e18) >= 20e9:
            picked.append(c)
            last[c["session"]] = c["ts"]
    rng.shuffle(picked)
    picked = [c for c in picked if cv2.imread(c["frame"]) is not None][:a.n]      # a few recorder JPEGs are cut
    os.makedirs(a.out, exist_ok=True)
    cells = [cell(cv2.imread(c["frame"]), c["box"], n, ctx=3.0, colour=(0, 200, 255)) for n, c in enumerate(picked)]
    write_sheets(cells, a.out, "lead")
    with open(os.path.join(a.out, "lead_sample.json"), "w") as f:
        json.dump(picked, f)
    print("lead candidates", len(cands), "picked", len(picked), file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("dets"); d.add_argument("scan"); d.add_argument("out")
    d.add_argument("--per-class", type=int, default=45); d.add_argument("--min-h", type=float, default=14); d.add_argument("--seed", type=int, default=42)
    l = sp.add_parser("lead"); l.add_argument("sessions", nargs="+"); l.add_argument("--out", required=True)
    l.add_argument("--n", type=int, default=150); l.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    {"dets": cmd_dets, "lead": cmd_lead}[a.cmd](a)


if __name__ == "__main__":
    main()
