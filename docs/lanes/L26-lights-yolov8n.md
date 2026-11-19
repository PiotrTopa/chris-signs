# L26: BDD100K YOLOv8n as the traffic-light detector, compared with the YOLO11n in use

Lane of the roadsense 2.0 campaign, 2026-11-18 23:44 to 2026-11-19 ~01:45 CET (Opus). Laptop only; the phone
was not touched. Branch `v2-l26` of roadsense-signs (from `v2` b781a8b). Scratch: `~/datasets/l26` (models,
replay outputs, ncnn build) and `/mnt/win/linux-scratch/l26` (ride frames, car scans, calibration images).
Times CET.

Question: does `dronefreak/bdd100k-yolov8n` give the same light quality as `dronefreak/bdd100k-yolo11n`
(the installed detector, L7), and is it cheaper in a form the phone can use (every layer on Vulkan; int8 on the
CPU)? Rules: the guidelines in roadsense `docs/WAYS_OF_WORKING.md` (same quality is measured after the trackers).

## Verdict: reject (no change to the default; no new option added)

* After the tracker, on the three replay rides, the YOLOv8n reports less of the light-time than the YOLO11n:
  97.0 vs 97.4 % by day, 95.3 vs 98.6 % in the night tunnel, 92.9 vs 98.9 % in the night snow, and it loses the
  far lights (under 12 px: 77 vs 82 % by day, 82 vs 94 % at night). It reports fewer lights with no reference
  box near (33 vs 51 per minute by day) and flickers no more. Under guideline 3 that is not the same quality.
* On BDD100K val (the L7 scoring) it is better in the wide view (AP50 62.9 vs 61.1, +1.8, 95 % paired bootstrap
  +0.9..+2.9) and slightly worse in the far view (49.7 vs 50.4, -0.7, -1.7..+0.1). The gain on US dashcam
  pictures does not carry over to the Polish rides after the tracker.
* On the car's own recordings it finds 87.7 % of the lit lights the YOLO11n found at the start threshold 0.25
  (98.5 % at the keep threshold 0.12); boxes only one of them finds are mostly not lights on both sides.
* It is not cheaper where the phone has room. It has about 30 % more arithmetic (6.5 vs 4.9 GFLOPs at 960x320,
  scaled from the cards' 8.7 / 6.6 at 640). Laptop CPU: ORT fp32 91 vs 82 ms, ncnn fp32 95 vs 99 ms, ncnn int8
  77 vs 84 ms (2 threads, interleaved, shared machine). Every one of its 212 ncnn layers has a Vulkan kernel
  (the YOLO11n has two MatMuls without one), but the GPU is 91 % busy in time (L21). A light detector on the
  GPU costs road runs whether or not it falls back to the CPU in between.
* int8 does not rescue it: with the best recipe found (below) it loses 1.2 AP50 in the wide view and 2.4 in
  the far view (YOLO11n: 0.6 / 1.3), and after the tracker it is worse again (night tunnel 88.9 %, far lights
  56 %).

Freed budget: none.

## 1. Checkpoint and its confinement

* `https://huggingface.co/dronefreak/bdd100k-yolov8n/resolve/2c24162504d096e7bb0a12ec0af6bfeb9d5d697a/best.pt`,
  sha256 `acd17780b331c3970e22faf3c6f123eae8b8c4abcc736ad5ef38ff28ecd72fb2` (`args.yaml` 3bb2de77...,
  `README.md` 808b915a...). `args.yaml`: `model: yolov8n.pt`, 960 px, 50 epochs (same recipe as the YOLO11n).
  `tools/scan_pt.py`: 23 globals, none outside the allowlist.
* Loaded once (owner's approval of 2026-11-18, this checkpoint and this evaluation only) to export static ONNX
  at 320x960 (the lights input) and 736x1280, opset 17, no NMS, fp32, through Ultralytics 8.4.190 (`~/l7/venv`).
  Confinement: `systemd-run --user -p MemoryMax=8G -p CPUQuota=200% --wait --pipe` running `bwrap` with
  `--ro-bind / /` (whole filesystem read-only), empty tmpfs over `/home`, `/tmp`, `/run`, `/media`, `/mnt`,
  `/srv`, `/var/tmp`; visible on top of that only the venv (`/home/piotr/l7/venv`, read-only), the export
  script directory (read-only) and one work directory (`~/datasets/l26/export`, the only writable place);
  `--unshare-all` (own network namespace with only a down loopback: a test connect gave `Network is
  unreachable`; own pid, ipc, uts, user, cgroup namespaces), `--die-with-parent --new-session --clearenv`.
  Checked before the load with a shell in the same box: `/home/piotr` held only `l7/venv`, `~/.ssh` absent,
  writes outside the work directory failed. The Windows disk opened later at `/mnt/win` was under the `/mnt`
  tmpfs as well.
* Then the `.pt` copies were deleted (`~/datasets/l26/export/v8n.pt`, `~/datasets/l26/dl/best.pt`); all later
  work used the ONNX. Class list identical to the YOLO11n's (8 = traffic light). An older, never loaded copy
  with the same sha256 still sits in `~/datasets/l7-models/bdd100k-yolov8n/best.pt` (L7 reported it deleted;
  not removed here, it is not this lane's file).

| file (local only, research licence) | sha256 |
|---|---|
| `bdd100k_yolov8n_320x960.onnx` (onnxsim) | `fe3862e0...` |
| `bdd100k_yolov8n_320x960-ncnn.{param,bin}` | `79cdf79c...` / `312e0989...` |
| `bdd100k_yolo11n_320x960.onnx` (= `~/l7/models/bdd11n_320x960.onnx`) | `8a694cfc...` |
| `bdd100k_yolo11n_320x960-ncnn.bin` | byte-identical to L7's |

## 2. BDD100K val (L7 method)

`eval_det.py` / `score_det.py` of L7 (copies with only the output path changed), the 2000 images with
sha1 % 5 == 0 (5,533 lights), ORT fp32, same crops as L7. The YOLO11n row reproduces L7's 61.1 exactly.

| detector | view (input 960x320) | AP50 | AP30 | R / P @0.25 | R h<12 | 12-24 | 24-48 | >=48 px |
|---|---|---|---|---|---|---|---|---|
| YOLO11n | wide: rows 0-427 @0.75 | 61.1 | 78.2 | 56.0 / 75.6 | 20.8 | 49.4 | 74.5 | 80.9 |
| YOLOv8n | wide | **62.9** | 79.0 | 56.8 / 76.3 | 23.6 | 50.8 | 73.8 | 80.7 |
| YOLO11n | far: 960x320 at 1.0 (x 160-1120, y 90-410) | **50.4** | 67.0 | 49.9 / 70.5 | 35.6 | 51.3 | 56.6 | 40.5 |
| YOLOv8n | far | 49.7 | 66.4 | 48.4 / 69.6 | 33.4 | 50.2 | 54.8 | 39.3 |
| YOLO11n int8 (aciq, head fp32) | wide | 60.5 | 77.4 | 56.2 / 74.7 | 20.7 | 49.6 | 74.9 | 80.7 |
| YOLOv8n int8 (aciq, head fp32) | wide | 61.7 | 78.3 | 54.8 / 76.3 | 24.9 | 48.7 | 70.7 | 78.2 |
| YOLO11n int8 (aciq, head fp32) | far | 49.1 | 66.4 | 49.2 / 68.9 | 33.8 | 50.5 | 56.7 | 39.3 |
| YOLOv8n int8 (aciq, head fp32) | far | 47.3 | 65.7 | 45.9 / 67.8 | 32.2 | 46.3 | 53.6 | 38.7 |
| YOLO11n int8 (kl, all convs) | wide / far | 46.0 / 34.1 | | 36.0 / 65.5 | 7.4 | | | |
| YOLOv8n int8 (kl, all convs) | wide / far | 44.0 / 31.6 | | 34.1 / 64.7 | 6.2 | | | |

Paired bootstrap over images (300 resamples), YOLOv8n minus YOLO11n: wide AP50 +1.8 [+0.9, +2.9], recall under
12 px +2.8 [+0.6, +5.1]; far AP50 -0.7 [-1.7, +0.1], under 12 px -2.2 [-4.8, +0.3]. The two models' scores are
calibrated alike (precision at 0.25: 76.3 vs 75.6), so both run with the same thresholds; nothing was tuned.
The 736x1280 full-frame runs were stopped to free the shared machine (context only, not needed for the verdict).

## 3. After the tracker: replay rides

`lights_replay` (this branch, ORT fp32, 1 thread) on the first 5 minutes (frames 0-5999) of `krakow-lights`
(day), `krakow-night-tunnel`, `gdansk-night-snow`; ROIs per ride as in L15/L21. Reference per detector as in
L15 (four views, every 4th frame, no tracker, thr 0.2, BDD state model); scored with `tools/lights/replay_score.py`
against the **union** of both detectors' references (8 files), so neither model is its own judge. Runs: the
v2 unit's configuration `P5` (5 Hz = every 4th frame, `--dt-ms 200 --coast 3 --thr-keep 0.12`, second view,
car state model) and the 3 Hz configuration of L15 (`P3`, every 7th frame, coast 2). State: share of the covered
reference light-frames reported in the chain's majority colour (`tools/lights/replay_state.py`, new).

| ride (union reference: lit lights, light-s) | run | lights found | light-time | < 12 px | 12-24 px | drop-outs / min | extra / min | state same / other / off |
|---|---|---|---|---|---|---|---|---|
| krakow-lights, day (48, 882 s) | YOLO11n P5 | 47 | **97.4 %** | **82.1 %** | 97.7 % | 0.3 | 51.2 | 65.1 / 13.1 / 21.8 % |
| | YOLOv8n P5 | 47 | 97.0 % | 77.0 % | 97.9 % | 0.1 | **32.6** | 65.6 / 13.0 / 21.5 % |
| | YOLO11n int8 P5 | 46 | 97.6 % | 85.6 % | 97.5 % | 0.4 | 70.0 | 64.8 / 12.5 / 22.7 % |
| | YOLOv8n int8 P5 | 46 | 96.4 % | 72.5 % | 97.5 % | 0.3 | 34.4 | 69.0 / 14.4 / 16.6 % |
| | YOLO11n P3 | 45 | 94.3 % | 67.6 % | 94.0 % | 2.0 | 27.0 | 65.5 / 13.4 / 21.1 % |
| | YOLOv8n P3 | 47 | 94.0 % | 62.6 % | 94.7 % | 1.8 | 20.0 | 68.3 / 14.0 / 17.7 % |
| krakow-night-tunnel (28, 85 s) | YOLO11n P5 | 28 | **98.6 %** | **94.3 %** | 99.6 % | 0.0 | 19.4 | 99.0 / 0.5 / 0.5 % |
| | YOLOv8n P5 | 27 | 95.3 % | 81.6 % | 98.5 % | 0.0 | 17.0 | 99.5 / 0.0 / 0.5 % |
| | YOLO11n int8 P5 | 28 | 96.7 % | 85.1 % | 99.6 % | 0.0 | 22.4 | 99.0 / 0.5 / 0.5 % |
| | YOLOv8n int8 P5 | 26 | 88.9 % | 56.3 % | 97.4 % | 0.0 | 11.6 | 99.5 / 0.0 / 0.5 % |
| | YOLO11n P3 | 26 | 91.3 % | 77.0 % | 95.9 % | 2.8 | 6.4 | |
| | YOLOv8n P3 | 26 | 89.2 % | 63.2 % | 98.5 % | 2.1 | 6.2 | |
| gdansk-night-snow (12, 36 s) | YOLO11n P5 | 12 | **98.9 %** | 100 % (16) | 99.1 % | 1.6 | 30.0 | 98.9 / 0 / 1.1 % |
| | YOLOv8n P5 | 11 | 92.9 % | 100 % | 89.6 % | 1.6 | 11.8 | 98.8 / 0 / 1.2 % |
| | YOLO11n int8 P5 | 12 | 98.4 % | 100 % | 99.1 % | 0.0 | 15.4 | 100 / 0 / 0 % |
| | YOLOv8n int8 P5 | 12 | 97.8 % | 100 % | 99.1 % | 1.6 | 28.8 | 100 / 0 / 0 % |

With each detector's own reference instead of the union (P5, YOLO11n vs YOLOv8n light-time): YOLO11n
reference 98.3 / 97.9 % by day, 98.4 / 95.2 % tunnel, 99.0 / 89.0 % snow; YOLOv8n reference 98.0 / 97.7 %,
100 / 99.6 %, 99.2 / 100 %. Only the snow ride (7 lights) turns over when the YOLOv8n judges itself; full output
in `~/datasets/l26/scripts/ride_scores*.txt`. "extra" counts reported light-frames with no reference box of
either detector within 0.35 s: coasting tails and weak keep-threshold boxes, as in L15. The state column depends
on the state model, not the detector: the same within 1-4 points. The night rides are small (85 and 36
light-seconds): directions, not precise numbers; the day ride carries the weight.

## 4. Car recordings

`tools/lights/car_scan.py` with each detector (both unit views, thr 0.12, car state model) on the four recorder
sessions L21 scanned (`20261117-101013`, `-145921`, `-220714`, `20261118-095209`; 9,157 frames, one a second).

* L21's labelled sample (320 YOLO11n detections at >= 0.25, labelled by eye): the YOLOv8n has a box near 87.7 %
  of the 195 lit lights at >= 0.25 and 98.5 % at >= 0.12; near 68 % (>= 0.25) of the 125 non-lights / back
  views. The sample was drawn from the YOLO11n's detections, so it can only show what the YOLOv8n loses.
* Boxes at >= 0.25 that only one detector has (no box of the other at >= 0.12 near): YOLO11n 219 of 2,165,
  YOLOv8n 242 of 2,005. A seeded sample of 80 per side, labelled by eye from contact sheets
  (`/mnt/win/linux-scratch/l26/car/cmp/only_*.png`): YOLO11n-only 2-4 lit lights and 1 dark housing; YOLOv8n-only
  6 lit lights (one uncertain red disc) and 9 dark or back housings; the rest on both sides are not lights (the
  same pole-mounted round back of a sign in front of a yellow house, about 30 frames each, windows, lit windows
  at night, sign backs). Scaled to the populations: about 5-11 lit lights only the YOLO11n has against about 18
  only the YOLOv8n has, while the labelled sample says the YOLOv8n misses about 12 % of the YOLO11n's lit
  lights at the start threshold (it keeps them at the keep threshold). Per frame on the car the two are close;
  the YOLOv8n starts tracks on fewer of the lit lights. One labeller (this lane), crops of 10-40 px.

## 5. Cost

| | YOLO11n | YOLOv8n |
|---|---|---|
| parameters / GFLOPs at 960x320 (card values scaled) | 2.6 M / ~4.9 | 3.0 M / ~6.5 |
| ncnn layers / without a Vulkan kernel | 295 / 2 (MatMul, C2PSA) | 212 / **0** |
| ncnn fp32 CPU vs ORT, max diff (one BDD image) | boxes 0.003 px, scores 0 | boxes 0.003 px, scores 0 |
| ncnn Vulkan fp16 (Quadro P3200) vs ORT | boxes 6.2 px, scores 0.009 | boxes 3.1 px, scores 0.005 |
| laptop ORT fp32 CPU, 2 threads, median (p10-p90) | 81.5 (71-92) ms | 91.1 (83-97) ms |
| laptop ncnn fp32 CPU, 2 threads | 98.6 (92-108) ms | 94.5 (87-102) ms |
| laptop ncnn int8 CPU (aciq, head fp32), 2 threads | 83.7 (81-92) ms | 77.3 (74-85) ms |
| laptop ncnn Vulkan fp16, Quadro P3200 | 22.8 (15-45) ms | 18.2 (12-39) ms |
| `lights_replay` ncnn-cpu int8 detector time on the laptop, median (P5 runs, 3 jobs in parallel) | 171 ms | 156 ms |

Laptop timings: `bench.py` (all eight cases interleaved, 30 rounds, random input), load average 9-13 on 12
threads from other lanes; relative only. The laptop's x86 int8 kernels are not the phone's ARM kernels (the
Snapdragon 845's A75 gains from int8 differently, and the phone runs the YOLO11n with fp16 arithmetic, which x86
does not have), so none of this is a phone number. On the phone: see the runbook (section 7).

int8 recipe (chosen on a disjoint BDD split, sha1 % 10 == 2, 1,000 images, wide view, then scored once on the
L7 split): calibration 400 BDD val images with sha1 % 5 == 1 (each with a light), cropped and resized to the
detector view. AP50 on the selection split, fp32 / kl / kl + head fp32 / aciq / aciq + head fp32: YOLO11n
62.2 / 46.8 / 58.1 / 58.3 / **61.7**; YOLOv8n 64.5 / 47.1 / 60.5 / 61.6 / **63.7**. "head fp32" leaves the
first conv and the head's final 1x1 convs unquantised (`tools/ncnn/int8_table_head_fp32.py`, new). ncnn's
default kl calibration on every conv is unusable for both (-15 AP50, small lights lost); YOLO11n's attention
MatMuls stay fp32 in ncnn int8 anyway, and the feared attention-block loss did not show.

## 6. What this means for L21's int8 proposal (YOLO11n)

The YOLO11n in int8 (aciq, head fp32) loses 0.6 / 1.3 AP50 on BDD and, after the tracker, is level by day
(97.6 vs 97.4 %, but 70 vs 51 extra reports per minute) and below at night (96.7 vs 98.6 %, under 12 px 85 vs
94 %) with the laptop's x86 int8 kernels. Not shown to be the same quality; whether it is cheaper on the A75s is
unmeasured. If a lane times it on the phone, it must repeat the tracker-level check with the phone's own
backend (`lights_replay --backend ncnn-cpu` on the phone, as in L21).

## 7. Phone timing runbook (not run; the phone belonged to other lanes)

Files (laptop, local only, never commit or re-host; BDD100K / AGPL terms as in `docs/LIGHT_MODELS.md`):

```sh
# laptop
M=~/datasets/l26/models
ssh user@PHONE mkdir -p l26/models l26/f l26/out
scp $M/bdd100k_yolo11n_320x960-ncnn.{param,bin} $M/bdd100k_yolov8n_320x960-ncnn.{param,bin} \
    $M/bdd100k_yolo11n_320x960-int8aciqhead-ncnn.{param,bin} $M/bdd100k_yolov8n_320x960-int8aciqhead-ncnn.{param,bin} \
    user@PHONE:l26/models/
ls /mnt/win/linux-scratch/l26/frames/krakow-lights/*.jpg | awk 'NR%4==1' | head -300 | xargs -I{} scp {} user@PHONE:l26/f/
```

Detector alone, the installed backend and placement (2 threads on the big cores, fp16 arithmetic), with the
roadsense stack stopped or in the same state for every row, each model twice in alternating order:

```sh
# phone (lights_replay of v2 or later: /usr/local/bin/lights_replay)
cd ~/l26
for rep in 1 2; do for m in yolo11n yolov8n yolo11n-int8 yolov8n-int8; do
  case $m in *-int8) P=models/bdd100k_${m%-int8}_320x960-int8aciqhead-ncnn.param ;; *) P=models/bdd100k_${m}_320x960-ncnn.param ;; esac
  lights_replay --backend ncnn-cpu --model $P --threads 2 --cpus 4-7 --no-track --state rule \
      --roi 0,0.261,1,0.854 --bench 2 f/*.jpg > /dev/null 2> out/cpu-$m-$rep.log
  # Vulkan (GPU must not be shared with modeld for a clean number; the YOLO11n row runs its MatMuls on the CPU)
  case $m in *-int8) ;; *) lights_replay --backend ncnn --model $P --threads 2 --cpus 4-7 --no-track --state rule \
      --roi 0,0.261,1,0.854 --bench 2 f/*.jpg > /dev/null 2> out/vk-$m-$rep.log ;; esac
done; done
grep -h "median/p90" out/*.log
```

In the live stack (cost against road runs, power, temperatures): L21's harness on the phone,
`sudo sh ~/l21/run.sh NAME ~/l21/profiles/NAME.sh 240 60`, with a profile that copies the proposed stack's
`SIGND_ARGS` (roadsense `tools/budget/stack.sh`) and adds `--lights-model /home/user/l26/models/<file>.param`;
rows: YOLO11n (as installed), YOLOv8n ncnn-cpu, YOLO11n int8, YOLOv8n int8. Compare `sg-lights` big-core % and
`lights ms (detector)` from the daemon stats. Any int8 row then needs the tracker-level check on the phone:
`lights_replay --backend ncnn-cpu --model <int8 param> --threads 2 --dt-ms 200 --coast 3 --thr-keep 0.12
--roi ... --roi2 ... --state <car state model>` on the frames of section 3 and `tools/lights/replay_score.py`
against this lane's union references (`~/datasets/l26/rides/*.ref_*.jsonl`).

## 8. Reproduce, files left

Scripts (scratch, not in the repo): `export_once.py` + `box.sh` (confined export), `eval_det.py` /
`eval_det_ncnn.py` / `score_det.py` / `boot.py` (BDD), `conv.sh` (ONNX -> ncnn, the `make_ncnn.sh` steps),
`cmp_ncnn.py` (Vulkan placement), `make_calib.py`, `int8.sh` / `int8b.sh` (calibration, variants, selection),
`rjobs.sh` / `rrun.sh` (ride replays), `car_compare.py`, `bench.py`; copies in `~/datasets/l26/scripts`.
In the repo (this branch): `tools/lights/replay_state.py`, `tools/ncnn/int8_table_head_fp32.py`, this report,
one paragraph of `docs/LIGHT_MODELS.md`.

Left on the laptop: `~/datasets/l26` (models incl. int8 variants and tables, replay outputs,
BDD outputs, `ncnn/` source + build with `ncnn2table` / `ncnn2int8` / `benchncnn`, `build/` = this branch's
`lights_replay` with ncnn), `/mnt/win/linux-scratch/l26` (ride frames, car scans with crops, contact sheets, calibration
images). No `.pt` of this lane remains.

Not verified: any phone number (time, power, heat, fp16 parity of the YOLOv8n on the Adreno or the A75); int8
on ARM; night and rain beyond two short night rides; the ego-light replay (`lights_ego_replay.py`) was not run.
