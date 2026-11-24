# roadsense-signs

Traffic-sign provider for [roadsense](https://github.com/PiotrTopa/chris-roadsense):
`signd` reads camera frames from roadsense's shared-memory frame ring, runs a
class-agnostic YOLOv8n sign detector (384x640) and a 400-class Mapillary (MTSD)
YOLOv8n-cls classifier on each box with ONNX Runtime on the CPU (or ncnn on the
GPU through Vulkan, `--backend ncnn`), tracks signs with a SORT-style tracker,
and publishes `signs` messages on a Unix socket that the roadsense api picks up.
With `--lights` a second thread also detects traffic lights (BDD100K YOLO11n on the
upper part of the frame, a small state CNN, a tracker) and publishes `lights`
messages on `<rundir>/lightd.sock`; models and licences in
[docs/LIGHT_MODELS.md](docs/LIGHT_MODELS.md), offline runs with `lights_replay`.
With `--lights-road-users` the same detector's road users (person, rider, car, truck,
bus, train, motor, bike) go out in every `lights` message as well, from the inference
that already runs for the lights (roadsense lane L42; section "Road users" of
docs/LIGHT_MODELS.md); `--lights-ru-roi` adds an optional extra detector run on a lower
band of the picture for the near road users.
`--lights-ru-scale` adds to each road user the image scale change (`scale_rate`,
`scale_rate_std`, `scale_dt_s`: d ln(size)/dt from patch alignment against the same object
0.2-1 s earlier, for range rate and time to contact; section "Image scale change" of
docs/LIGHT_MODELS.md; checks: `build/scale_rate_test`, `tools/road_users/scale_check.py`).
Detection inputs for a tracker (all off by default, each adds only optional keys;
[docs/DETECTION_INPUTS.md](docs/DETECTION_INPUTS.md)): `--lights-ru-low X` (the boxes below the
class thresholds as a list of their own, `road_users_low`, for a second association stage),
`--lights-ru-flags` (`trunc`, `occ`, `occ_by`, `occ_edges`: cut by the frame, covered by a nearer
box), `--lights-ru-view` (`sym`, `lamps`: mirror symmetry, the lamp pair of a vehicle in the dark),
`--lights-ru-emb yuv|FILE.onnx` (`emb`: an appearance vector for re-identification),
`--lights-ru-scale-lamps` / `--lights-ru-scale-multi N` (the scale rate in the dark from the lamp
pair's separation; several baselines). Checks: `build/detin_test`, `build/scale_rate_test`,
`tools/road_users/detin_bdd.py`, `tools/road_users/low_check.py`, `tools/reid/`.

Split out of roadsense at commit
[`246e68a`](https://github.com/PiotrTopa/chris-roadsense/commit/246e68a) (where it
lived as `signd/`, `tools/signd_replay.cc` and `docs/SIGN_MODELS.md`), renamed
from pocopilot to roadsense on the way.

## Licence

AGPL-3.0, see [LICENSE](LICENSE). The daemon code is ours, but the only sign
models fast enough for the target hardware are Ultralytics-trained (Ultralytics
claims AGPL-3.0 over trained weights) and these particular weights carry no
licence at all (all rights reserved; trained on MTSD, CC BY-NC-SA), so the whole
sign stack lives here, under AGPL-3.0, for private non-commercial use, and
roadsense (MIT) stays free of it. The weights are never committed or re-hosted:
`scripts/fetch-models.sh` downloads the pinned `.pt` files and exports them
locally. Details in [docs/SIGN_MODELS.md](docs/SIGN_MODELS.md).

## How it plugs into roadsense

Purely over roadsense's documented IPC contract (roadsense
`docs/ARCHITECTURE.md`); nothing is linked:

* input: the frame ring `/dev/shm/roadsense-frames` (env `ROADSENSE_RING`),
  read with roadsense's header-only `camerad/ring.h`;
* output: `<rundir>/signd.sock` (rundir `/run/roadsense`, env
  `ROADSENSE_RUNDIR`), newline-delimited JSON via `camerad/jsonsock.h`; best
  crops as `<rundir>/crops/<track_id>.jpg` via `camerad/jpeg_nv12.h`;
* config: `$ROADSENSE_CONFIG_DIR` (default `/etc/roadsense`)`/signd.json`,
  models in `/var/lib/roadsense/models`.

If signd is not running the roadsense api reports it as not connected and
everything else keeps working. The systemd unit joins `roadsense.target`
(`PartOf=`, `WantedBy=`) without the target depending on it, and runs in
roadsense's `roadsense.slice` (`Slice=`) so the host that starts and stops the
target (on the Prius car computer: the car backend, on ACC) can bound the whole
stack's CPU and memory as one group.

### The `signs` message (one per processed frame)

```
{"type":"signs","seq":1203,"ts_mono_ns":...,"ts_real_ns":...,
 "detect_ms": 85.0,
 "detections": [ {"track_id": 17, "cls": "regulatory--maximum-speed-limit-50", "cls_id": 23, "score": 0.81,
                  "box": [x0,y0,x1,y1], "age": 12, "hits": 11} ],
 "tracks_ended": [ {"track_id": 15, "cls": "...", "first_seq": 1100, "last_seq": 1190, "hits": 40,
                    "best_score": 0.93, "best_crop": "/run/roadsense/crops/15.jpg"} ]}
```

`box` is in pixels of the full camera frame (1920x1080 on the reference phone);
`seq`/`ts_*` are the frame ring's. Tracks are reported in `detections` once
confirmed (`min_hits` 3) and end after `max_age` 10 processed frames without a
match. The api serves them at `/v1/signs`, `/v1/signs/history` and draws them on
`/preview`. Round red tracks (prohibitions, speed limits, stop) also carry `glow` / `verify` and, when signd
judges them to be a stop lamp or traffic light (verifier CNN, glow, per-class minimum score), are reported
flagged with `"suppressed": true, "reason": "..."` instead of being dropped
([docs/SIGN_MODELS.md](docs/SIGN_MODELS.md) section 6).

## Numbers (Poco F1, Snapdragon 845)

ONNX Runtime CPU (2026-10-21):

| | |
|---|---|
| detector + classifier, 4x Cortex-A75 | 114 ms cold, 145-180 ms sustained |
| same, 4x Cortex-A55 | ~320 ms |
| live in the full stack (signd on the A55s, roadsense modeld on the A75s) | 1.5-2 frames/s |

ncnn/Vulkan on the Adreno 630 next to roadsense modeld on the same GPU (2026-10-24, 70 s live runs
on the 1280x720 DVR camera; details in roadsense `docs/lanes/L1-signs-vulkan.md`):

| signd configuration | signs fps | road (modeld) runs/s | hottest zone after 70 s |
|---|---|---|---|
| ORT, 4 threads, A75s (old unit) | 5.6 | 9.7 | 94 C and rising |
| detector ncnn + classifier ORT, cpus 4-7, uncapped | 8.7 | 8.7 | 72 C |
| same, `--max-hz 6` / `--max-hz 4` | 6.0 / 4.0 | 8.7 / 9.6 | 69 / 66 C |
| both models on ncnn, cpus 0-3 | 8.1 | 8.1 | 71 C |

Accuracy of the fp16-packed GPU path vs ORT fp32: identical detections (21/21 boxes within 2 px,
scores within 0.01, same class) on three photos and 60 drive frames; classifier probability
differs by up to 0.23 on one small far sign. The shipped unit uses the detector-on-GPU split
capped at 5 frames/s.

Quality: every near/mid-range sign in six Polish street photos detected and
classified correctly (incl. yellow Polish warning signs); misses signs under
~15 px; one low-confidence false positive (docs/SIGN_MODELS.md, section 3).

## Build and install

Needs a roadsense checkout for its IPC headers (`camerad/ring.h`,
`camerad/jsonsock.h`, `camerad/jpeg_nv12.h`), ONNX Runtime, CMake and a C++17
compiler; libjpeg optional (JPEG input for `signd_replay`).

```sh
cmake -S . -B build -DROADSENSE_DIR=../roadsense      # default ROADSENSE_DIR is ../roadsense
cmake --build build -j
build/signd_replay --help                             # offline: images / NV12 dumps -> signs JSON
```

Reference deployment (Poco F1 / postmarketOS, next to an installed roadsense):

```sh
scripts/fetch-models.sh        # dev box: download pinned .pt, export ONNX + class lists into models/
git ls-files -z | tar czf - --null -T - | ssh user@phone 'mkdir -p ~/roadsense-signs && tar xzf - -C ~/roadsense-signs'
scp models/mtsd_* user@phone:~/roadsense-signs/models/
ssh user@phone 'cd ~/roadsense-signs && sudo sh scripts/install-phone.sh && sudo systemctl start roadsense-signd'
```

### GPU backend (ncnn / Vulkan)

`-DROADSENSE_NCNN=ON -DNCNN_DIR=/path/to/ncnn` (an ncnn source tree built in
`NCNN_DIR/build`, roadsense `scripts/build-ncnn.sh`) builds the ncnn runner in;
`scripts/fetch-models.sh` also converts the two ONNX models to
`models/*-ncnn.{param,bin}` (`tools/ncnn/make_ncnn.sh`: onnxsim, pnnx, and
`fix_ncnn.py`, which splits the detector's final Concat because ncnn computes it
wrongly), and `NCNN_DIR=... scripts/install-phone.sh` installs them. Then
`signd --backend ncnn` runs both models on the GPU, `--det-backend ncnn
--cls-backend ort` only the detector (the 64x64 classifier is faster on the
CPU). Precision is fp16-packed storage with fp32 arithmetic, the same as
roadsense modeld on the Adreno 630/turnip, including `TU_DEBUG=flushall`
(`--ncnn-no-flushall` to skip). The ORT path stays the reference:
`tools/ncnn/compare_models.py` (laptop) and `tools/ncnn/compare_replay.py` (two
`signd_replay --raw` outputs) measure the difference. Measurements and the
configuration decision: roadsense `docs/lanes/L1-signs-vulkan.md`.

Layout: `signd/` (daemon: `yolo.*` pre/post-processing, `runner.h` +
`ort_runner.cc` / `ncnn_runner.cc` inference backends, `tracker.*`, `output.*`
message + crops, `main.cc`), `tools/signd_replay.cc`, `tools/ncnn/` (conversion
and comparison), `tools/scan_pt.py` + `tools/export_onnx.py` (model export),
`systemd/`, `scripts/`, `docs/SIGN_MODELS.md` (model survey, licences,
pre/post-processing spec).
