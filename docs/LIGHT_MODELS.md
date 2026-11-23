# Traffic-light models for `signd --lights`

Date: 2026-10-27 (roadsense lane L7). Measurements, the cost table and the live checks are in roadsense
`docs/lanes/L7-traffic-lights.md`; this file records what the models are, where they come from and
under which terms.

## Pipeline

1. **Region**: the upper part of the frame (`--lights-roi 0,0,1,0.62`: lights sit above the horizon; on the
   1280x720 ring that is rows 0..446), resampled into the detector input `--lights-in 960x320` (scale 0.72),
   content top-left, grey padding.
2. **Detector**: `bdd100k_yolo11n_320x960` = YOLO11n fine-tuned on BDD100K (10 road classes; class 8
   `traffic light` is the only one used), Ultralytics head `[4+10, N]`, score > 0.25, NMS 0.45.
3. **State**: per box a 48x24 crop (horizontal housings rotated, left lamp on top) into a small CNN
   (`lights_state_bdd_48x24.onnx`, 4 logits red / yellow / green / none, ~1 ms per crop on ORT, one A75
   thread). Without that file the lamp colour rule in `signd/lights.cc` decides (weaker, see below).
4. **Tracker** (`LightTracker`): greedy IoU then centre-distance association, constant-velocity box
   centre; the state switches after two agreeing observations (or one at p >= 0.9); `red_yellow` is a
   yellow observation straight after red (Vienna-convention sequence); `flashing` = yellow and off
   alternating within 4 s.

Since 2026-11-16 (roadsense lane L15, measurements in roadsense `docs/lanes/L15-stability.md`), all three
optional and off by default:

* **Second view** (`--lights-roi2 x0,y0,x1,y1`): every other processed frame feeds this part of the frame to
  the same detector instead of the wide one. The unit uses the middle 960x320 pixels around the horizon at
  scale 1.0: far lights are 30 % larger for the detector there than in the wide view (scale 0.72), at the
  same number of detector runs. The tracker does not count a frame against a track that lies outside the
  view of that frame. `roi` and `view` in the message say which view a frame used.
* **Keep threshold** (`--lights-thr-keep 0.12`): boxes scoring between this and `--lights-thr` may continue
  an existing track (IoU, or a centre distance of 0.75 box sizes) and never start one.
* **Coasting reports** (`--lights-coast N`): a confirmed track is listed for N more processed frames after
  its last match, at its predicted box, with its last smoothed state and `coast` = frames since the match
  (`frame_state` is `unknown`). Without it a light vanished from the message whenever one frame missed it,
  and with two views it would vanish on every other frame.

Since 2026-11-20 (roadsense lane L48, measurements in roadsense `docs/lanes/L48-system-optimisation.md` section
3.2), off by default in the binary; **installed in the unit since 2026-11-21 (roadsense lane L53, owner's
decision): `--lights-idle ZW.Z.W`**:

* **Idle pattern** (`--lights-idle P`): while the tracker holds no track at all, the frame periods of
  `--lights-hz` follow the pattern `P` instead of wide, zoom, wide, ...: one letter per period, `W` = wide pass,
  `Z` = second view, `.` = no detector pass and no message. The first box that may start a track brings the full
  schedule back with the next frame. `ZW.Z.W` (a third of the idle passes saved) scored inside the method's
  noise on the two reference rides; every-other-period patterns did not. `<rundir>/demand` with
  `"lights_idle":0` (roadsense docs/ARCHITECTURE.md, "Demand hints") forbids the pattern while valid.
  `lights_replay --idle P` runs the same schedule offline; `--view-phase 1` gives the other phase of the normal
  schedule as a noise estimate.
* **Idle announcement** (since L53): every `lights` message sent while the pattern runs and nothing is tracked
  carries `"idle": {"pattern": "ZW.Z.W", "hz": 3.33}` (passes a second kept at the current `--lights-hz`, after the
  governor's cap). A consumer judges the stream's rate by it: chris-integrator's `lights` floor is then
  0.8 x `hz` instead of 3 a second, so an idle stream that keeps its pattern is not `slow` and one that falls
  behind it is. Messages without `idle` come from the full schedule.
* **Sparser patterns** (L53, laptop, the L48 method: `replay_score.py` on the first 5 minutes, every 4th frame,
  against the L15 reference; noise = the normal schedule started on the other view: light-time 0.0 points by day
  and by night, 0.7 at 24-48 px, first report 0.40 / 0.50 s):

  | idle pattern | idle passes saved | light-time day | 24-48 px day | first report day | light-time night |
  |---|---|---|---|---|---|
  | none (wide, zoom, ...) | 0 | 97.1 % | 94.4 % | 0.40 s | 98.9 % |
  | `ZW.Z.W` (installed) | 1/3 | 97.0 % | 93.3 % | 0.50 s | 99.5 % |
  | one pass in 2 (`W.Z.` / `Z.W.`) | 1/2 | 96.8 / 96.7 % | 92.7 / 91.7 % | 0.50 / 0.40 s | 97.9 / 98.9 % |
  | one pass in 3 (`W..Z..` / `Z..W..`) | 2/3 | 95.4 / 95.9 % | 86.0 / 87.4 % | 0.70 / 0.50 s | 97.9 / 96.3 % |
  | one pass in 4 (`W...Z...` / `Z...W...`) | 3/4 | 96.0 / 94.9 % | 88.4 / 82.5 % | 0.50 / 0.50 s | 94.4 / 97.1 % |

  Every pattern still reports every reference light at least once (27/27 by day, 16/16 by night). One pass in 2
  passes the coarse lines (light-time within 1.5 points, first report +0.1 s) but loses **1.7-2.7 points of the
  24-48 px (far) light-time against a noise of 0.7**: a measurable loss of far lights, so it is not installed;
  one in 3 and one in 4 lose light-time by 1-4.5 points. On the phone's own backend (ncnn CPU fp16) one in 2
  gives 96.6 % by day (24-48 px 91.7 against 93.6) and 97.4 % at night against 96.9 / 98.9 %; `ZW.Z.W` gives
  96.8 / 99.5 %. Measured with `ZW.Z.W` on the phone (rural ride without lights): light thread 88 -> 59 % of a
  big core, 0.41 W, 2 K on the big cores; with lights (krakow) 89 -> 76 %. Expected saving on idle roads of the
  sparser patterns, that figure scaled by the passes saved (inferred): one in 2 about 0.6 W, one in 3 about
  0.8 W, one in 4 about 0.9 W. Details: roadsense `docs/lanes/L53-owner-decisions.md`.

## Detector: dronefreak/bdd100k-yolo11n

| | |
|---|---|
| Source | https://huggingface.co/dronefreak/bdd100k-yolo11n, commit `2b5554133522f96f50ce87a4df0a2de556e4b7ad`, `best.pt` sha256 `c120d075bcc6aaf94523f9a2c9d681819ebd692f36eb5ea64656d1c5f1278a70` |
| Licence | model card: **AGPL-3.0** (Ultralytics); trained on **BDD100K**, whose licence allows research / non-commercial use. Same status as the MTSD sign models: private use, never commit or re-host the `.pt`, `.onnx` or ncnn files (`.gitignore` keeps them out); `scripts/fetch-models.sh` downloads the pinned `.pt`, scans its pickle globals (`tools/scan_pt.py`) and exports locally. |
| Training | DetectionBench recipe, 960 px, 50 epochs on BDD100K train; the card's "test" split is BDD100K's official val set (10 000 images), i.e. **held out**, which is what the evaluation below uses. Card: traffic light mAP50 64.2 at 960. |
| Export | `tools/export_onnx.py ... 320 960` (static, opset 17) -> `models/bdd100k_yolo11n_320x960.onnx`; ncnn: `tools/ncnn/make_ncnn.sh` (onnxsim, pnnx, the same head split as the sign detector: `out0` boxes, `out1` 10 class scores). ncnn has no Vulkan kernel for the two MatMuls of the C2PSA attention block, so on the GPU they run on the CPU between submits; on the CPU (`--lights-backend ncnn-cpu`) the whole graph runs with ncnn's ARM kernels. |
| Checks | laptop: ncnn CPU vs ORT max abs diff 0.002 px boxes, 0 scores; Vulkan fp16 (Intel) boxes <= 4 px, scores <= 0.015. Phone (C++ pipeline, NV12, ncnn-cpu fp16): AP50 61.9 vs 61.8 for the Python/ORT reference on the same 400 BDD images. |

Why this one (BDD100K val, 2000-image subset, 5533 lights, AP50 at IoU 0.5): COCO-trained YOLO11n finds
far fewer lights (AP50 10.4 at 640x384, 25.3 at 1280x736) than the BDD-trained one (49.6 / 67.8); there
is no permissive (Apache/MIT) dashcam traffic-light detector that runs on the phone (Autoware's Apache
YOLOX-s fine detector refines map ROIs of Japanese lights and is ~4x the compute (26.8 vs 6.5 GFLOPs at 640); COCO YOLOX-nano/tiny
are COCO-class, not evaluated after the COCO YOLO11n result). The BDD-trained YOLOv8n of the same recipe
(`dronefreak/bdd100k-yolov8n`, commit `2c24162`, all 212 ncnn layers on Vulkan) was compared in roadsense lane
L26 (`docs/lanes/L26-lights-yolov8n.md`) and rejected: +1.8 AP50 in the wide view on BDD100K val, but after
the tracker on the replay rides less light-time (97.0 vs 97.4 % by day, 95.3 vs 98.6 % and 92.9 vs 98.9 % at
night) and fewer far lights, with about 30 % more arithmetic; int8 does not change that.

## State CNN (trained here)

| | |
|---|---|
| Training | `tools/lights/train_state.py` on BDD100K val GT crops (>= 10 px tall), split by image (sha1 of the file name: even = train, odd = test), 40 epochs, inverse-frequency^0.85 class weights, brightness / mirror / shift augmentation. 4 conv blocks (16-32-64-64), ~65 k parameters. |
| Licence | the code is AGPL-3.0 with this repository; the **weights** are trained on BDD100K (research / non-commercial): local use only, never committed. `scripts/install-phone.sh` installs `models/lights_state_bdd_48x24.onnx` if it exists. |
| Accuracy | see roadsense `docs/lanes/L7-traffic-lights.md` (BDD test half and the Krakow ride). |

Alternatives measured on the same crops: the colour rule (hue + brightness + lamp position) and Autoware's
Apache-2.0 MobileNetV2 classifier (`AutowareFoundation/traffic_light_classifier`, 224x224, Japanese
horizontal lights; crops rotated so the red lamp is on the right) are both clearly worse than the small
CNN (numbers in the lane report); the rule mistakes blue sky behind a back view for a green lamp.
Red+yellow is not a BDD100K class and the rule cannot tell it from the glare of a red lamp, hence the
sequence inference in the tracker.

## State CNN for the car's own camera (`lights_state_car_48x24.onnx`, 2026-11-18)

Roadsense lane L21 (`docs/lanes/L21-budget-and-models.md` there). The same network, input and classes; what
changed is what it is trained on. On the first drives recorded by the car's camera (DVR camera, 1280x720
MJPEG, low sun) the 2026-10-27 model called 29 % of the lit lights `none`: against a bright sky that camera
shows a lit lamp as a white disc with a thin coloured rim on a dark housing, and nothing like it is in
BDD100K. Two steps, both in `tools/lights/`:

1. `train_state2.py`: BDD100K as before, with augmentations for what a dashcam and a detector do to a small
   lamp: box jitter, reduction to 7 px and back, veiling glare, exposure / gamma / channel gain, blur, JPEG,
   and **clipped highlights** (every pixel above a brightness threshold pushed towards white).
2. `finetune_state_car.py`: 12 more epochs on BDD100K with 30 % of every batch taken from labelled detections
   of the car's own drives.

The labelled sample: `car_scan.py` runs the detector (both views of the unit) and the state CNN on the
recorder's frames of two drives (2026-11-17, 29 and 70 min; 1,076 detections at score >= 0.25),
`car_sheets.py` draws a stratified sample of 320 for labelling by eye (half of them called `none` by the old
model), `car_state_eval.py` / `car_state_cpp.py` score a model in Python and through signd's own pipeline
(`lights_replay` on the phone). 195 of the 320 are lit lights, 125 are back or side views, dark or
unreadable housings and things that are no lights.

| model | lit lights of the car sample: right colour / wrong colour / `none` | others called lit | BDD100K test half, recall red / yellow / green / none |
|---|---|---|---|
| 2026-10-27 (`lights_state_bdd_48x24`) | 70.2 / 1.0 / 28.8 % (through signd on the phone: 69.7 / 1.0 / 29.3 %) | 14.2 % | 87.1 / 61.0 / 95.8 / 88.2 |
| augmentations without clipped highlights | 69.4 / 3.1 / 27.5 % | 14.5 % | 82.9 / 75.2 / 95.6 / 86.2 |
| step 1, with clipped highlights (no car picture in training) | 83.8 / 1.0 / 15.2 % | 15.4 % | 81.0 / 76.0 / 95.2 / 85.2 |
| step 1 + 2, cross-validated (five folds of one-minute blocks, a light's approach never on both sides) | **90.8 / 0.5 / 8.7 %** | **10.7 %** | 85.0-86.7 / 60.2-63.4 / 95.0-95.3 / 89.6-90.2 |
| the same fine-tuning without car crops (control) | 73.6 / 1.0 / 25.4 % | 9.8 % | 86.4 / 62.6 / 95.4 / 89.5 |
| step 2 alone on the 2026-10-27 model, cross-validated | 89.7 / 0.0 / 10.3 % | 11.6 % | 87.9-88.9 / 52.0-54.9 / 94.9-95.1 / 89.0-89.6 |
| installed file: step 1 + 2 on all 320 labels | in-sample (99.5 %; through signd 100 %): not a measurement | | 86.7 / 64.2 / 94.9 / 89.9 |

What this does and does not show. The clipped-highlight augmentation alone halves the lit lights reported
dark with no picture of the car in training, but it was designed after looking at the car's crops. The
cross-validated row is the honest estimate for drives like the two it comes from (one camera, two days,
sunny, low sun, day and dusk); night, rain and yellow phases (4 in the sample) are not in it. One labeller,
crops of 10-30 px, so the labels themselves carry errors: a crop was labelled lit only when the lamp was
visible. Arrows are still not a state. Reproduce:

```sh
python tools/lights/car_scan.py SCAN.npz ~/roadsense-rec/20261117-101013 ~/roadsense-rec/20261117-145921 ...
python tools/lights/car_sheets.py SCAN.npz SHEETS --n 320 --skip <parked sessions>    # then SHEETS/labels.txt by eye
python tools/lights/train_state2.py SAMPLES.json BDD_ROOT aug2w --epochs 60 --whiteout 0.5
python tools/lights/finetune_state_car.py aug2w.pt SAMPLES.json BDD_ROOT SCAN.npz SHEETS car   # -> car_48x24.onnx
python tools/lights/car_state_eval.py SCAN.npz SHEETS old.onnx car_48x24.onnx
```

Licence as for the first model: BDD100K (research / non-commercial) plus the owner's own frames; the weights
and the labelled sample stay local. `signd --lights-state A.onnx,B.onnx` takes the first file that exists, so
the unit names the car model first and the BDD model as the one to fall back to.

## Road users from the same detector (`signd --lights-road-users`, 2026-11-19)

Roadsense lane L42 (`docs/lanes/L42-road-users.md` there). The detector above is trained on all ten BDD100K
classes; the light thread used only class 8. With `--lights-road-users` it also decodes classes 0-7 (person,
rider, car, truck, bus, train, motor, bike) from the same output, one class per box (the best of the eight,
class-agnostic NMS 0.5, per-class thresholds `--lights-ru-thr`, default person 0.40, rider 0.35, car 0.40,
truck 0.40, bus 0.40, train 0.50, motor 0.35, bike 0.35, set before any labelling), pairs a rider with the bike
or motorbike under it into one road user, and lists them in every `lights` message as `road_users` (`cls`,
`score`, `box`, `clip`, `rider`, `p`). No extra detector run: the decode costs 0.04 ms median, 0.44 ms max per
light frame on the laptop (4500 frames of a ride, ncnn-cpu); the rest of the message is byte-identical with the
flag on or off. The load governor's first step (`road_users_hz` 0) pauses it (`"road_users": null`).

What the views see: the wide view ends at 62 % of the frame height, about 64 px below the car camera's horizon,
so a road user nearer than about 19 m is cut at the bottom (`clip` "b") and has no ground contact point for a
distance (the second view: nearer than about 40 m). `--lights-ru-roi x0,y0,x1,y1` (with `--lights-ru-hz`, default
1) adds an extra detector run on a lower band, road users only, after the light frame (a `road_users` message,
view 2); it costs one detector run each time and is off by default.

Accuracy on the car's camera, labelled by eye before scoring (`tools/road_users/sheets.py dets`, 216 boxes of
views 0 and 1, stratified by the detector's class, score >= 0.3, box height >= 14 px, the class not shown on the
sheets; one labeller; boxes of the same object in neighbouring seconds are not independent):

| label \ detector | bike | bus | car | motorbike | person | truck | dropped (below threshold) | n |
|---|---|---|---|---|---|---|---|---|
| bike (cyclist or bicycle) | 19 | | | | | | 3 | 22 |
| bus | | 35 | | | | 1 | 2 | 38 |
| car | | | 35 | 1 | | 6 | 11 | 53 |
| van | | 1 | 1 | | | 11 | 7 | 20 |
| truck | | | | | | 12 | 4 | 16 |
| tram | | 3 | | | | | 1 | 4 |
| person | 5 | | | | 38 | | 5 | 48 |
| other (no road user, part of one, unreadable) | 2 | | 5 | 1 | 2 | | 5 | 15 |

Precision of the kept boxes: person 0.95, bus 0.90 (trams called bus), car 0.88 (a van counted right as car:
the detector has no van class), bike 0.73 (most bikes are parked bicycles; pedestrians beside bicycles called
bike), truck 0.40 (BDD100K's truck takes in the vans: 11 of 30 truck calls are vans, 6 are cars), motorbike 0/2.
No `train` box in 12 576 frames. Reproduce: `tools/road_users/car_scan.py`, `sheets.py dets`, `score.py`.

## Image scale change per road user (`signd --lights-ru-scale`, 2026-11-23)

With `--lights-road-users --lights-ru-scale` every road user may carry `scale_rate` (1/s, d ln(size)/dt of the
object's IMAGE over the baseline, positive = growing = approaching), `scale_rate_std` (1/s, 1 sigma) and
`scale_dt_s` (s, the baseline); the three keys come together and are absent when not measured. For an object
ahead, closing speed / distance = `scale_rate` (time to contact 1 / `scale_rate`), independent of the object's
true size, the camera pitch or the road slope (Stein, Mano, Shashua 2003; Dagan et al. 2004), and without the
box edges' jitter (3 px, 10-15 % of a far car's height) that a rate from box heights carries. Code:
`signd/scale.h`, `signd/scale.cc`; off by default (deployment: add the flag to the unit).

How (per light pass, any view; boxes are full-frame pixels, so wide-view and zoom-view boxes of one object compare
directly):

* **Patches from the full-resolution luma** of the ring frame (1280x720), not from the detector's downscaled view:
  a far car keeps its native pixels. Template = the central 80 % of the box, about 24x24 template pixels
  whatever the aspect (long side up to 96 so a flat strip, e.g. a near car cut at the view's bottom, keeps its 8
  rows); boxes up to ~28 px take raw integer-aligned pixels, larger ones are area-averaged to <= 0.66 scale.
  At most 12 boxes a pass (largest first, `--lights-ru-scale-max`).
* **Association** without a tracker: a box continues the chain (same class group: person/rider/bike/motor or
  car/truck/bus/train) whose box, predicted with constant centre velocity and log-size rate, overlaps it best
  (IoU >= 0.25); chains end after 1.2 s without a box.
* **Baseline**: the chain's oldest patch 0.2-1.0 s back (`--lights-ru-scale-dt`); at the unit's schedule the
  median baseline is 0.8 s.
* **Alignment**: the current frame is resampled around the box at the box-ratio scale prior (a raw crop for
  native templates, so the warp interpolates once and no image of the pair is blurrier than the other: double
  interpolation biased far objects by -0.007/s in the first version); NCC over log scale (step 1/template
  radius, 0.04-0.08, range +-0.08..0.28 from the box jitter) and integer shift (+-2..6, all shifts at the prior
  scale, +-2 around that peak at the others); then Gauss-Newton on log scale, shift, gain and offset.
* **Uncertainty**: residual variance x (J^T J)^-1, with the gradients' own noise removed from J^T J (errors in
  variables), inflated by the residual's lag-1 autocorrelation per axis ((1+rho)/(1-rho)).
* **No value** when the box touches the frame edge or is clipped on both sides of an axis by the view, the
  template is under 8 px on a side or textureless (sd < 2.5 grey levels), the search image leaves the frame, the
  NCC peak is < 0.7 or on the edge of the search range, the fit leaves the peak or does not converge, the fit's
  R^2 < 0.6, or the stated 1-sigma of ln(scale) exceeds 0.06. A box clipped by the view on one side gets its
  scale prior from the free axis and its position from the free edge.

### Synthetic check (`build/scale_rate_test`, ctest `scale_rate`)

Textured rectangle growing at a known rate over a static textured background, 4x4 supersampled, Gaussian noise
3 grey levels, box edges jittered 2 px (4 px for the near case), 5 passes a second for 1.2 s, 60 random textures
per case, values from 0.4 s on (dt 0.4-1.0 s):

| case | coverage | bias (1/s) | scatter (1/s) | stated std (1/s) | rms pull | > 3 sigma | ms/box (laptop) |
|---|---|---|---|---|---|---|---|
| far 16 px, still | 87 % | +0.0003 | 0.0163 | 0.0126 | 1.04 | 0.0 % | 0.12 |
| far 16 px, closing 0.15/s, moving | 89 % | -0.0027 | 0.0138 | 0.0108 | 1.20 | 1.9 % | 0.14 |
| far 24 px, contrast 15, closing 0.10/s | 82 % | +0.0035 | 0.0223 | 0.0199 | 1.09 | 1.2 % | 0.29 |
| mid 60 px, closing 0.30/s | 100 % | -0.0003 | 0.0029 | 0.0029 | 0.93 | 1.0 % | 0.37 |
| mid 60 px, opening 0.20/s | 100 % | -0.0003 | 0.0030 | 0.0038 | 0.76 | 0.3 % | 0.37 |
| near 180 px, closing 0.40/s | 100 % | +0.0002 | 0.0018 | 0.0019 | 0.86 | 0.3 % | 0.37 |
| uniform (textureless) 60 px | 0 % | | | | | | |
| cut by the frame edge | 0 % | | | | | | |
| tiny 6 px | 0 % | | | | | | |

Bias stays below a fifth of the scatter; the stated std is within 0.75-1.2 of the actual scatter.

### Recorded data (`lights_replay --road-users --ru-scale --name-fps F`, ORT, `tools/road_users/scale_check.py`)

The car's recorder sessions of 2026-11-22 (night, city, 1280x720 at **one frame per 1.03 s**, ego mean 3.7
m/s) are too sparse for this: run with `--ru-scale-dt 0.2,1.6`, half the objects appear in one frame only or
move too far in a second to be re-associated (13-14 % of 297 wide-view boxes measured), and successive estimates
1 s apart of near leads in city traffic differ by real dynamics (14 pairs, rms z 3.7). The leads that held their
distance read about 0 (rate x distance / ego speed in [-0.25, 0.25] for 13 of 16 boxes with ego > 3 m/s); no
parked car passed at speed got two frames, so the v / x relation could not be checked. **A recording with
every frame (or 5 a second) is needed** for an honest car-camera check: `roadsense-rec` with a 5 Hz frame rate
on a day and a night drive with parked cars along the road.

The ride library at 20 fps, cut to the unit's schedule (5 frame periods a second, wide / zoom alternating, idle
pattern ZW.Z.W, `--lights-roi2 0.125,0.125,0.875,0.569`), frame times from the file names:

| ride | boxes | measured | far boxes (size < 40 px) measured, median stated std | successive estimates, disjoint baselines |
|---|---|---|---|---|
| warsaw-city (day, 15 min, stop-and-go) | 1368 | 613 (45 %) | 57 of 159, 0.0086 /s (12-20 px: 79 %) | far: 51 pairs, rms z 1.28, 3.9 % > 3; both rates < 0.05: 447 pairs, rms z 1.49, 4.0 % > 3 |
| krakow-night-tunnel (night, 5 min) | 195 | 70 (36 %) | 59 of 140, 0.046-0.056 /s | far: 48 pairs, rms z 0.68, 0 % > 3 |
| a1-night (night motorway, 5 min) | 33 | 7 (21 %) | 7 of 23, 0.034-0.051 /s | 6 pairs, rms z 0.57 |
| krakow-lights (day, 15 min) | 108 | 23 (21 %) | none | 18 near-object pairs, rms z 4.7 (dynamics) |

Why boxes get no value (warsaw-city, 1368 boxes): first sighting of the object 287, box at the frame edge 175,
clipped on both sides 62, no earlier patch in the window 63; of the 781 boxes that reached the alignment 613
were measured (78 %), the rest failed NCC 35, fit 69, search edge 14, R^2 12, std 13, tiny 8, frame edge 17.

Consistency of successive estimates is a noise check only where the true rate changes slowly: every pair
beyond 3 sigma among the "slow" warsaw pairs is a car 4-6 m ahead (box 330-400 px wide, cut at the view's
bottom) in a queue starting or stopping, where 0.25 m/s^2 changes the rate by 0.05/s within a second.

Against modeld's lead on warsaw-city (lead prob > 0.5; our box in the path whose width matches a 1.8 m car at the
lead's distance, 445 boxes): our value minus (v_ego - v_lead) / x_lead has median -0.0005/s, MAD 0.0021/s;
examples x 6.4 m, model -0.323 vs ours -0.332 +- 0.018; x 6.1 m, -0.097 vs -0.107 +- 0.009; stopped leads
model ~0 vs ours +-0.002 +- 0.005. (Ride views end above these cameras' horizon, so the boxes have no ground
distance of their own and the parked-car v / x check needs the car's camera.)

Against box differencing over the same baseline (same objects, the box-size ratio of the same two passes):
the box rate scatters around the image rate by 0.065/s (far boxes < 40 px), 0.039/s (40-80 px), 0.014/s
(> 80 px) on warsaw-city, where the image rate states 0.009, 0.009 and 0.005/s; at night (krakow-night-tunnel,
far boxes) 0.096/s vs a stated 0.049/s. So by day a far object's rate is about 7x more precise from one pair of
patches than from its box; at night (headlight-lit, taillights, low texture) about 2x.

Cost (laptop, i7 class, one thread; per light pass with road users): median 0.001 ms (nothing to measure), p90
0.7 ms, max 2.4 ms over 4500 warsaw-city frames; per measured box 0.12 ms (16 px) to 0.37 ms (60-180 px), so
the 12-box cap bounds a pass at about 4.5 ms. On the phone's big cores (3-5x slower) about 0.4-1.9 ms per box,
p90 2-4 ms and worst case 13-22 ms a pass at the cap; on a little core 10x. Memory: <= 64 chains x ~7 patches of
<= ~1000 floats. The measurement runs after the detector, outside the GPU turn, and its time is in `lights_ms`
(and `timing.scale_ms` with `--lights-timing`). No new dependencies.

Reproduce: `lights_replay --backend ort --model bdd100k_yolo11n_320x960.onnx --roi2 0.125,0.125,0.875,0.569
--thr-keep 0.12 --coast 3 --idle ZW.Z.W --road-users --ru-scale --name-fps 20 <ride frames 0,4,8,...>.jpg >
out.jsonl` (reject counts on stderr), then `tools/road_users/scale_check.py out.jsonl --road
<rides-road>/<ride>.road.jsonl.gz`; recorder sessions: `--name-fps 19.47 --ru-scale-dt 0.2,1.6`, then
`scale_check.py out.jsonl --session <session dir>`.
