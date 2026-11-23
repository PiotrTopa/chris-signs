#!/usr/bin/env python3
"""scale_check.py OUT.jsonl [--session DIR [--fps F]] [--road RIDE.road.jsonl.gz] [--outliers] [--label NAME]:
truth-free checks of the road-user scale rate (signd --lights-ru-scale) in lights_replay output
(--road-users --ru-scale --name-fps F).

  coverage by box size, distribution of scale_rate and its stated std
  successive estimates of one object (boxes linked offline by overlap, disjoint baselines) agree within their
    stated std, split by size and by rate (a fast-changing rate is real dynamics, not noise)
  --session DIR (roadsense recorder session: frames/<camera seq>.jpg, messages.jsonl.gz with ego speed and calib;
    --fps = the camera's rate, which --name-fps was given): static objects passed at ego speed v show v/x
    (x = forward ground distance from the box bottom); histogram of rate * x / v (1 static, 0 same speed)
  --road FILE (ride library: modeld road messages at 20 Hz, v_ego and leads): our value on the box of modeld's
    lead (in the path, width of a 1.8 m car at the lead's distance) vs (v_ego - v_lead) / x_lead
Needs chris-integrator (camera model): CHRIS_INTEGRATOR=/path (default ../../../chris-integrator from here).
Lane: roadsense-signs docs/LIGHT_MODELS.md "Image scale change".
"""
import json, gzip, math, sys, bisect, statistics as st

import os
sys.path.insert(0, os.environ.get("CHRIS_INTEGRATOR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../chris-integrator")))
from chris_integrator.core.roadusers import Camera, project  # noqa: E402

args = sys.argv[1:]
out = args[0]
session = args[args.index("--session") + 1] if "--session" in args else None
calib_path = args[args.index("--calib") + 1] if "--calib" in args else None
road_path = args[args.index("--road") + 1] if "--road" in args else None
fps = float(args[args.index("--fps") + 1]) if "--fps" in args else 19.47
label = args[args.index("--label") + 1] if "--label" in args else out

frames = []
for l in open(out):
    m = json.loads(l)
    frames.append(m)

cam = None
speed_t, speed_v, cam_ts = [], [], {}
if session:
    calib = None
    for l in gzip.open(session + "/messages.jsonl.gz", "rt"):
        m = json.loads(l)
        t = m.get("type")
        if t == "calib" and calib is None:
            calib = m
        elif t == "ego" and m.get("speed_mps") is not None:
            speed_t.append(m["ts_mono_ns"] * 1e-9)
            speed_v.append(m["speed_mps"])
        elif t == "camera":
            cam_ts[m["seq"]] = m["ts_mono_ns"] * 1e-9
    cam = Camera(calib)
elif road_path:  # ride: modeld's road messages (v_ego, leads) at 20 Hz, frame index i <-> ts i * 50 ms
    leads = {}
    for l in gzip.open(road_path, "rt"):
        m = json.loads(l)
        if m.get("type") == "calib":
            cam = Camera(m)
        elif m.get("type") == "road" and m.get("v_ego") is not None:
            t = m["ts_mono_ns"] * 1e-9
            speed_t.append(t), speed_v.append(m["v_ego"])
            ld = [q for q in m.get("leads") or [] if q.get("t_offset") == 0]
            if ld and ld[0]["prob"] > 0.5:
                leads[round(t * 20)] = (ld[0]["x"][0], ld[0]["v"][0], m["v_ego"])
    session = "ride"
    cam_ts = None
elif calib_path:
    c = json.load(open(calib_path))
    cam = Camera(c)


def mean_speed(t0, t1):
    i, j = bisect.bisect_left(speed_t, t0), bisect.bisect_right(speed_t, t1)
    v = speed_v[i:j]
    return st.mean(v) if v else None


boxes = []  # (frame index, ts, obj)
for fi, m in enumerate(frames):
    for o in m.get("road_users") or []:
        boxes.append((fi, m["ts_mono_ns"] * 1e-9, o))

n = len(boxes)
meas = [b for b in boxes if "scale_rate" in b[2]]
print("== %s: %d frames, %d boxes, %d with scale_rate (%.0f %%)" % (label, len(frames), n, len(meas), 100.0 * len(meas) / max(1, n)))


def h(o):
    return o["box"][3] - o["box"][1]


print("  coverage by box size sqrt(w h) (px): bin  boxes  measured  median |rate|  median std  p90 std")
sz = lambda o: math.sqrt((o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]))
for lo, hi in [(0, 12), (12, 20), (20, 40), (40, 80), (80, 2000)]:
    bb = [b for b in boxes if lo <= sz(b[2]) < hi]
    mm = [b[2] for b in bb if "scale_rate" in b[2]]
    if not bb:
        continue
    sd = sorted(o["scale_rate_std"] for o in mm)
    print("    %4d-%-4d %6d %6d (%3.0f%%)  %8s  %8s  %8s" % (lo, hi, len(bb), len(mm), 100.0 * len(mm) / len(bb),
          "%.4f" % st.median(abs(o["scale_rate"]) for o in mm) if mm else "-",
          "%.4f" % st.median(sd) if mm else "-", "%.4f" % sd[int(0.9 * (len(sd) - 1))] if mm else "-"))
if meas:
    r = sorted(b[2]["scale_rate"] for b in meas)
    q = lambda a, p: a[int(p * (len(a) - 1))]
    print("  scale_rate 1/s: p5 %.3f p25 %.3f median %.3f p75 %.3f p95 %.3f; dt median %.2f s" % (
        q(r, .05), q(r, .25), q(r, .5), q(r, .75), q(r, .95), st.median(b[2]["scale_dt_s"] for b in meas)))

# --- link boxes offline: greedy IoU between consecutive frames that hold boxes (same class group)
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


grp = lambda c: 0 if c in ("person", "rider", "motor", "bike") else 1
tracks = []  # list of lists of (ts, obj)
live = []
for fi, m in enumerate(frames):
    ts = m["ts_mono_ns"] * 1e-9
    objs = m.get("road_users") or []
    live = [t for t in live if ts - t[-1][0] < 1.7]
    pairs = sorted(((iou(o["box"], t[-1][1]["box"]), i, j) for i, o in enumerate(objs) for j, t in enumerate(live)
                    if grp(o["cls"]) == grp(t[-1][1]["cls"])), reverse=True)
    used_o, used_t = set(), set()
    for s, i, j in pairs:
        if s < 0.2 or i in used_o or j in used_t:
            continue
        live[j].append((ts, objs[i]))
        used_o.add(i), used_t.add(j)
    for i, o in enumerate(objs):
        if i not in used_o:
            t = [(ts, o)]
            tracks.append(t)
            live.append(t)

# successive measured estimates with non-overlapping baselines: (r1 - r2) / sqrt(s1^2 + s2^2 (+ s1 s2 when they share
# the frame between them: its noise enters both with opposite signs))
z, zs = [], []
for t in tracks:
    mm = [(ts, o) for ts, o in t if "scale_rate" in o]
    for k in range(len(mm)):
        for k2 in range(k + 1, len(mm)):
            t1, o1 = mm[k]
            t2, o2 = mm[k2]
            gap = (t2 - o2["scale_dt_s"]) - t1  # start of the later baseline minus end of the earlier one
            if gap < -0.02:
                continue
            s1, s2 = o1["scale_rate_std"], o2["scale_rate_std"]
            shared = abs(gap) < 0.02
            d = math.sqrt(s1 * s1 + s2 * s2 + (s1 * s2 if shared else 0))
            z.append((o2["scale_rate"] - o1["scale_rate"]) / d)
            zs.append((sz(o1), o1, o2, t1, t2))
            break  # the next disjoint one only
if z:
    za = sorted(abs(v) for v in z)
    print("  successive estimates of one object (disjoint baselines): %d pairs, rms z %.2f, median |z| %.2f "
          "(0.67 for N(0,1)), |z|>3: %.1f %%" % (len(z), math.sqrt(sum(v * v for v in z) / len(z)), st.median(za),
                                                  100.0 * sum(1 for v in za if v > 3) / len(za)))
    def zrep(name, sel):
        if sel:
            print("    %-36s %4d pairs, rms z %.2f, median |z| %.2f, |z|>3 %.1f %%" % (name, len(sel), math.sqrt(sum(v * v for v in sel) / len(sel)),
                  st.median(abs(v) for v in sel), 100.0 * sum(1 for v in sel if abs(v) > 3) / len(sel)))
    zrep("size < 40 px (far)", [v for v, q in zip(z, zs) if q[0] < 40])
    zrep("size >= 40 px", [v for v, q in zip(z, zs) if q[0] >= 40])
    zrep("both |rate| < 0.05 (slow geometry)", [v for v, q in zip(z, zs) if abs(q[1]["scale_rate"]) < 0.05 and abs(q[2]["scale_rate"]) < 0.05])
    zrep("some |rate| >= 0.05", [v for v, q in zip(z, zs) if not (abs(q[1]["scale_rate"]) < 0.05 and abs(q[2]["scale_rate"]) < 0.05)])

# --- static-object check (needs ego speed and calibration)
if session and cam:
    rows = []
    for fi, ts, o in boxes:
        if "scale_rate" not in o:
            continue
        seq = int(round(ts * fps))
        tm = cam_ts.get(seq) if cam_ts is not None else ts
        if tm is None:
            continue
        v = mean_speed(tm - o["scale_dt_s"], tm)
        p = project(o, cam)
        if v is None or p["x_m"] is None:
            continue
        x = p["x_m"]
        rows.append((o, v, x, p))
    print("  static check: %d measured boxes with a ground distance and ego speed" % len(rows))
    mv = [r for r in rows if r[1] > 3.0]
    if mv:
        ratio = sorted(r[0]["scale_rate"] * r[2] / r[1] for r in mv)
        hist = {}
        for q_ in ratio:
            b = round(q_ * 4) / 4
            hist[b] = hist.get(b, 0) + 1
        print("    ego > 3 m/s: %d boxes; ratio rate*x/v histogram (1 = static, 0 = same speed as us, >1 oncoming):" % len(mv))
        for b in sorted(hist):
            if -1.5 <= b <= 3:
                print("      %+5.2f %s" % (b, "#" * hist[b]))
        print("      outside [-1.5, 3]: %d" % sum(c for b, c in hist.items() if not -1.5 <= b <= 3))
    for o, v, x, p in sorted(rows, key=lambda r: -r[2])[:40]:
        print("    %-6s h %5.1f px  x %5.1f m (y %+5.1f)  v %4.1f m/s  v/x %.3f  rate %+.4f +- %.4f  dt %.2f" % (
            o["cls"], h(o), x, p["y_m"], v, v / x, o["scale_rate"], o["scale_rate_std"], o["scale_dt_s"]))

if "--outliers" in args and z:
    print("  outlier pairs (|z| > 3):")
    for v, (hh, o1, o2, t1, t2) in sorted(zip(z, zs), key=lambda q: -abs(q[0]))[:25]:
        if abs(v) <= 3:
            break
        print("    z %+6.1f  %s t %.1f->%.1f  box1 %s  r1 %+.4f+-%.4f  r2 %+.4f+-%.4f  box2 %s" % (
            v, o1["cls"], t1, t2, [round(x) for x in o1["box"]], o1["scale_rate"], o1["scale_rate_std"], o2["scale_rate"],
            o2["scale_rate_std"], [round(x) for x in o2["box"]]))

if road_path and cam:
    # the modeld lead (prob > 0.5): our measured box nearest the path column whose width matches a 1.8 m wide car at
    # the lead's distance within a factor 1.4 (ride views end above the horizon: no ground distance of our own)
    pc = cam.cx
    rows = []
    for fi, m in enumerate(frames):
        ts = m["ts_mono_ns"] * 1e-9
        L = leads.get(round(ts * 20))
        if not L or L[0] < 3:
            continue
        wexp = cam.f * 1.8 / L[0]
        best = None
        for o in m.get("road_users") or []:
            if "scale_rate" not in o or o["cls"] not in ("car", "truck", "bus"):
                continue
            w = o["box"][2] - o["box"][0]
            cxo = 0.5 * (o["box"][0] + o["box"][2])
            if not (wexp / 1.4 < w < wexp * 1.4) or abs(cxo - pc) > 0.6 * w + 40:
                continue
            if best is None or abs(cxo - pc) < abs(0.5 * (best["box"][0] + best["box"][2]) - pc):
                best = o
        if best:
            rows.append((best, L))
    print("  lead check vs modeld (lead prob > 0.5; box in the path, width = 1.8 m at the lead's distance +-40 %%): %d boxes" % len(rows))
    if rows:
        d = [o["scale_rate"] - (L[2] - L[1]) / L[0] for o, L in rows]
        md = st.median(d)
        print("    ours - (v_ego - v_lead)/x_lead: median %+.4f, MAD %.4f; |ours| median %.4f; model's own value median |.| %.4f" % (
            md, st.median(abs(x - md) for x in d), st.median(abs(o["scale_rate"]) for o, L in rows),
            st.median(abs((L[2] - L[1]) / L[0]) for o, L in rows)))
        for o, L in rows[:: max(1, len(rows) // 20)]:
            print("    x_lead %5.1f m  v_ego %4.1f v_lead %4.1f -> %+.4f   ours %+.4f +- %.4f  box w %.0f" % (
                L[0], L[2], L[1], (L[2] - L[1]) / L[0], o["scale_rate"], o["scale_rate_std"], o["box"][2] - o["box"][0]))
