# Pose of a road user from a trained head (`signd --lights-ru-pose FILE.onnx`)

Date: 2026-11-24 (roadsense lane `viewhead`). The detection inputs of [DETECTION_INPUTS.md](DETECTION_INPUTS.md)
left two things open (its section 6): which way a vehicle points, by day, and where it meets the road. Both need
a trained model. This file is signd's half: how a pose head is run, what is published, what was measured through
signd's own code path, and what it costs. The model itself (training, data, licence, the full evaluation) is a
separate project, **chris-pose**; the message contract is in chris-roadsense `docs/ARCHITECTURE.md` (light provider);
how a tracker should take the fields: chris-roadsense `docs/lanes/detin-consumption-design.md`, section 9.

**Off by default.** With the option off the messages are byte for byte what they were (checked: every message
line of a day and a night ride at the unit's schedule, 1500 passes each, with the detection inputs off and with
all of them on, against the outputs of commit 5948fea, the time fields set aside). `ctest` (6 tests) passes
without any model file.

## 1. The model file is not part of this repository

signd is AGPL-3.0. The head is trained on the KITTI data set, whose licence (CC BY-NC-SA 3.0) forbids commercial
use; a non-commercial restriction cannot be put on AGPL code, so the trained weights are **separately
distributed data, loaded at run time** from the path given on the command line. This repository holds only the
generic machinery (`signd/pose.h`, `signd/pose.cc`, the option in `LightPipeline`) and a stand-in model for the
tests that is a fixed linear map written out in `tools/pose/make_fixture.py` (no training, no data). It contains
no KITTI-derived weights, labels, pictures or tables; the numbers quoted below are measurements.

* Where the weights come from: chris-pose, release artefact `pose1-m64.onnx` (sha256 in its model card),
  CC BY-NC-SA 3.0 with KITTI's attribution. On the laptop: `<datasets>/viewhead-lane/out/`.
* What that means: fine on a private, non-commercial car; the file must not be bundled into a commercial
  product; a commercially usable head needs retraining on permissively licensed or synthetic data (the training
  code is MIT). The owner decides what ships. This is the project's reading, not legal advice.
* A missing file: one line on stderr, no `pose` keys, everything else as usual.

## 2. The contract `pose1`

Any model with this input and output can be loaded (the runner checks the shapes).

* **Input** `[1, 12, 64, 64]`. Planes 0-2: the box scaled by 1.3 about its centre, stretched to 64 x 64, RGB / 255
  (signd's NV12 resampler, the same as for the other heads), black where the crop leaves the picture. Planes
  3-11: nine numbers about the box, each as a constant plane (so that the model has one input, like every model
  signd runs): ln(width / height); four flags: the left / top / right / bottom edge is cut (`clip`, `trunc`, or
  within 2 px of the frame's edge); 1 when the camera's numbers are given (`--lights-ru-pose-cam F,CU,CV`: focal
  length and principal point in frame px), else 0 and the next three 0; (box centre x - cu) / f; (box bottom - cv)
  / f; ln(box height / f) + 3.
* **Output** `[1, 38]`, finished numbers (the model carries its own calibration): heading, its 1-sigma, the two lobe
  probabilities, four view-class probabilities, seven ground-point coordinates relative to the box with their
  1-sigma, four "seen" probabilities, three dimensions with their 1-sigma, occlusion, truncation, three kind
  probabilities. The layout is in `signd/pose.h`; `ru_pose_decode` turns box units into frame pixels.
* **Heading** `yaw` (degrees): against the line of sight from the camera, clockwise seen from above. 0 = the object
  points away (rear seen), +-180 = it points at the camera, +90 = it points to the right (its right side is seen).
* **Ground points**: `gc` the centre of the footprint (the rectangle length x width under the object); `gn` where
  the line of sight to the centre enters that rectangle (straight ahead: the middle of the rear face on the road;
  the near face a tracker's ground row is about), the same column as `gc`; `gl` / `gr` the leftmost / rightmost
  bottom corner on the sides that face the camera. The points are amodal: where the bottom is cut or covered the
  head extrapolates and says so (`seen` low, sigma large).

## 3. What signd does

* `--lights-ru-pose PATH.onnx` (signd.json `lights_ru_pose`; `lights_replay --ru-pose`), with
  `--lights-road-users`. One ONNX Runtime run per box on one thread, after the flags, the view cues and the
  appearance vector, inside the light thread; nothing touches the GPU turn.
* **Which boxes**: vehicles (car, truck, bus, train) of the first list by size, then of the low list.
  `--lights-ru-pose-all` adds persons and two-wheelers; it is off because the head's heading for them is not
  usable (section 4), so by default a person or a two-wheeler has no `pose`: unknown, not guessed. A box under 8 px
  on a side gets none.
* **Cap**: `--lights-ru-pose-max N` boxes a pass (default 4). With more boxes than that the passes take turns
  through the list (pass k starts where pass k - 1 stopped), so every box gets a pose every few passes; a tracker
  must expect `pose` to be absent on some boxes.
* **Camera numbers**: `--lights-ru-pose-cam F,CU,CV` hands the focal length and principal point to the head.
  Measured on the training set's own camera they do not help (section 4), and the head has seen only that
  camera's geometry: leave the option unset.
* **Message**: per road user `pose` (keys in chris-roadsense `docs/ARCHITECTURE.md`), per message `pose_kind` "pose1";
  with `--lights-timing` the timing gains `pose_ms`. The governor's `road_users_hz` 0 sheds it with the road
  users.

## 4. Measured

The model: chris-pose `pose1-m64` (495 k parameters, 16.0 M multiply-accumulates per crop), trained on KITTI's
training drives. Full tables: the model card of chris-pose; here the figures a user of the option needs.

### Against labels (KITTI's held-out drives, the detector's own boxes)

12651 vehicles of drives the model never saw; boxes = this detector's (first and low list, the 320x960
export on the whole KITTI picture), matched to labels at IoU 0.5; camera numbers not given.

| heading | median error | over 30 deg | flipped (over 90 deg) |
|---|---|---|---|
| all vehicles | 2.5 deg | 8.8 % | 6.2 % |
| seen from behind / in front | 1.7 / 2.0 deg | 6.3 / 3.5 % | 4.9 / 3.4 % |
| three-quarter views | 4.3 deg | 13.1 % | 8.9 % |
| side views | 14.2 deg | 36.9 % | 17.6 % |
| largely occluded | 3.5 deg | 13.5 % | 8.0 % |
| cut by the picture's edge | 4.8 deg | 22.2 % | 15.9 % |
| persons / two-wheelers (not published by default) | 23.5 / 47.9 deg | 43.6 / 57.1 % | 18.2 / 33.9 % |

The stated uncertainty: of the answers in the right lobe (92.5 %), 68.6 % lie within one `yaw_std`,
91.7 % within two, 2.8 % beyond three (heavy tails); `flip` and `p` are honest to a few points over
their whole range (stated `flip` 0.03 / 0.14 / 0.27 / 0.41: about that share is flipped). Over time (held-out
tracking sequences): half of the in-lobe error is the object's own offset (sigma 2.4 deg), successive
errors correlate 0.65 at 0.2 s and 0.33 at 1 s; flips come and go (5.4 % of the boxes,
1.5 % of the tracks flipped in more than half of their boxes).

| ground points, visible uncut vehicles | error, median / 90 % (px) | in box units | the box alone (px) |
|---|---|---|---|
| `gc` column (the centre's bearing) | 1.51 / 7.04 | 0.021 widths | box centre: 2.97 / 17.72 |
| `gn` row (the near face on the road) | 1.28 / 6.52 | 0.028 heights | box bottom: 1.31 / 8.30 |
| `gl` / `gr` column | 2.16 / 1.87 (medians) | 0.029 / 0.025 widths | |

Stated 1-sigma of the points: 70.2 % of the `gn` rows within one, 93.3 % within two (68.4 /
89.8 % on occluded vehicles).

As a range (with the object's own ground height, so that only the pixel error counts; median / 90 % point of
the relative error): near face from `gn` 2.6 / 7.0 %, from the box bottom 2.8 /
8.2 %; nearer than 15 m 3.4 % against 5.0 %. **On a whole box `gn` is as good as
the detector's bottom edge, not better** (their errors correlate 0.72). What the head adds:

* **the centre**: the centre's range from `gc` is as good as the near face's (2.7 % median), where the box
  bottom is short by half the object's depth (8.0 %, biased -7.9 %); and the centre's bearing
  is twice as good as the box centre's;
* **a bottom that is not in the box**: with the lower 25-55 % of the box cut away (as the unit's views cut most
  near boxes: `clip` b), `gn` is found to 0.06-0.08 of the true box height in the median (0.16-0.18
  at the 90 % point), without bias, with a stated sigma on the safe side (80-92 % within one):
  for a car seen from 1.2 m up that is about 9 % of the range, where such a box gave a bearing only. Under a
  nearer labelled box: median 0.051, 90 % point 0.236 of the box height (stated sigma too small there:
  61.1 % within one);
* **the heading** at all.

Dimensions: for cars no better than the mean car (median |ln error| of the height 0.050 against 0.052);
for vans the detector calls `car`, 0.158 against 0.252 of the class prior; for lorries wrong
(0.382).

### Through signd's own code path

The same held-out KITTI pictures through `lights_replay --ru-pose` (signd's crop from the NV12 frame, ONNX Runtime)
against chris-pose's own evaluation path on the same boxes (`scripts/check_runtime.py`):
5792 vehicles with the same detector box in both: the heading differs by 0.17 degrees in the median (0.70 at the
90 % point; in 1.5 % of the boxes by more than 30 degrees: boxes on the edge of a flip, which a slightly different
crop tips), the `gn` row by 0.09 px (1.3 px at the 90 % point), `gc` by 0.19 px. Against the labels the two paths
are equal: heading median 2.47 / 2.48 degrees, over 30 degrees 8.7 / 8.8 %, `gn` row 1.72 / 1.73 px.

### Model size

Three sizes trained side by side on the same batches (same recipe), the detector's boxes, held-out drives:

| | crop | parameters | multiply-accumulates | heading: median, over 30 deg, flipped | `gn` row error, whole boxes (box heights) | `gn` with 40 % of the box cut (true box heights, median) |
|---|---|---|---|---|---|---|
| s48 | 48 x 48 | 438 k | 9.0 M | 2.6 deg, 9.5 %, 6.0 % | 0.041 | 0.075 |
| **m64 (shipped)** | 64 x 64 | 495 k | 16.0 M | 2.5 deg, 8.8 %, 6.2 % | 0.044 | 0.076 |
| w64 | 64 x 64 | 780 k | 33.3 M | 2.3 deg, 7.5 %, 4.9 % | 0.041 | 0.070 |

Twice the arithmetic buys 1.3 points of headings off by more than 30 degrees; a little over half of it costs 0.7
points and nothing on the ground points. The middle one is shipped; the small one is the candidate if the phone
needs it (it takes a 48 x 48 crop: a contract of its own). Side views swing between the three (22-37 % off by more
than 30 degrees): 601 boxes of a few tracks, too few to rank them by.

### On rides (other cameras, no labels)

Seven rides of the ride library (other cameras than KITTI's: dash cameras behind a windscreen, 1280 x 720, focal
lengths 690-1100 px, 1.05-1.7 m up; three by day: overcast city, rain, low sun; four at night: a lit tunnel and
expressway, a motorway, a motorway in fog, a snowy lit city), 5 minutes each, the whole frame through the
detector's 384x640 export at 5 passes a second, a pose on every vehicle box (`tools/pose/pose_check.py` of
chris-signs). None of these rides took part in training. **There are no labels**: each check holds the head against
another cue that has errors of its own.

| ride | vehicle boxes | red lamp pair: heading in the rear half (where the head is sure: `p` >= 0.8, `flip` < 0.1) | white lamp pair: heading in the front half (sure) | vehicle ahead (keeps its size 3 s in mid-picture): view rear | successive boxes 0.2 s apart: flipped | `gn` row - box bottom, box heights: median, spread |
|---|---|---|---|---|---|---|
| warsaw-city (day, overcast) | 6588 | 8: 100 % | 70: 24 % (19 %) *lamp rule wrong, below* | 13 chains, 983 poses: 97 %, median abs yaw 2.1 deg | 4.7 % | -0.010, 0.038 |
| katowice-rain (day, rain) | 4661 | 340: 91 % (97 %) | 567: 46 % (46 %) *lamp rule wrong, below* | 6 chains, 804: 92 %, 2.6 deg | 4.2 % | -0.023, 0.038 |
| s7-elblag-lowsun (day, low sun) | 1545 | none | 10: 0 % *lamp rule* | 4 chains, 460: 97 %, 2.3 deg | 1.4 % | -0.036, 0.044 |
| krakow-night-tunnel | 546 | 220: 97 % (99 %) | 138: 96 % (98 %) | none | 4.2 % | +0.012, 0.044 |
| a1-night-fog | 392 | 181: 93 % (100 %) | 8: 62 % | 2 chains, 103: 100 %, 0.8 deg | 7.6 % | +0.015, 0.051 |
| a1-night | 709 | 435: 95 % (100 %) | 19: 42 % | 3 chains, 187: 97 %, 2.9 deg | 7.3 % | +0.041, 0.054 |
| gdansk-night-snow (lit, snowy city) | 3296 | 434: 62 % (76 %) | 242: 45 % (34 %) | 2 chains, 247: 53 %, 11.5 deg | 11.2 % | -0.002, 0.039 |

What these say, and what they do not:

* **Heading by day**: the vehicle ahead is seen from behind in 92-99 % of its boxes (median |yaw| 2-3 degrees) on
  all three day rides: the common case works on cameras the head never saw. Crossing vehicles (chains that move
  sideways against the other boxes at a constant size: 11 chains, 115 poses on warsaw-city; 4 chains, 53 poses on
  gdansk-night-snow) point the way they move in 84 % and 94 % of the poses, but are called side-on (45-135 degrees)
  in only 42 % and 23 %: many are turning (three-quarter), and side views are also the head's weakest class on
  KITTI. Eleven chains prove little.
* **The lamp colour is a reference only in real darkness.** By day the lamp rule finds "white pairs" in dull
  frames (overcast, rain) on vehicles seen from behind; looked at by eye (sheets of 24 and 48 such boxes of
  warsaw-city and katowice-rain: all rear views, tail lamps visible) the head is right and the rule's colour wrong.
  A consumer must not take `lamps.c` white by day as "front"; with a pose it can tell.
* **Night, expressway and motorway**: red lamp pairs go with a rear-half heading in 93-97 % (99-100 % where the
  head is sure), white pairs with a front-half heading in 96 % on the one ride with enough of them (138); the two
  motorway rides have 8 and 19 white pairs (oncoming traffic across the median, small and glaring) and there the
  head is right in about half: not usable.
* **Night, snowy lit city: weak.** Against the lamp colour 62 % / 45 %, flips between successive boxes 11 %, the
  vehicle ahead called rear in 53 % of its boxes; and the head is surer than it should be (76 % right where it
  states `p` >= 0.8). By eye on a sheet of 48 lamp-pair boxes of that ride about 30 headings are clearly right
  and 5 clearly wrong, the rest cannot be judged, and several disagreements are the lamp rule's (parked vans
  facing the camera called red): both cues fail there.
* **What the lamp augmentation bought**: the same head trained without the drawn lamps called glowing tail lamps
  "front" with high confidence at night: on krakow-night-tunnel red pairs 71 % in the rear half, white 82 %,
  flips between successive boxes 14.2 %; with them 97 %, 96 % and 4.2 %. (On KITTI's day test the two heads are
  alike: 8.2 % and 8.8 % of the headings off by more than 30 degrees.) The check uses the lamp colour, the
  augmentation teaches lamp colour: it shows that the head now reads the lamps, not that it is right where the
  lamps mislead.
* **Ground contact**: on boxes with no cut and no covered edge the near-face row `gn` lies within 0.01-0.04 box
  heights of the detector's bottom edge in the median (spread 0.04-0.05), with a ride-dependent offset: in low sun
  `gn` is 3.6 % of the box height above the box bottom, on the dark motorway 4.1 % below. One of the two is biased
  there (a shadow thrown towards the camera, a bottom lost in the dark); without labels it cannot be said which.
  The centre `gc` is above `gn` in every box and inside the box in 84-99 %. That `gn` does not contradict the box
  where the box is good proves no accuracy: the gain claimed for `gn` is on boxes whose bottom is cut or hidden,
  and that is measured on KITTI only.
* **Over time**: the heading changes by 0.8-2.2 degrees in the median between boxes 0.2 s apart (KITTI, with
  labels: 1.3 degrees for a true change of 0.6), 0.13-0.38 of the joint stated sigma (KITTI: 0.45): the answers are
  smooth and, as on KITTI, far from independent. `gn` relative to the box moves by 0.010-0.022 box heights.
* The camera numbers of these rides (focal length, horizon) were used for range comparisons only, never given to
  the head.

## 5. Cost

Laptop (i7 class), one thread, **while other jobs loaded the machine** (load average 18-50 on 12 cores, as in
DETECTION_INPUTS.md section 7; a quiet laptop is faster by up to a factor of two). Phone: estimates with the
project's documented factors (big core 3-5 x a quiet laptop, little core about 10 x: LIGHT_MODELS.md). Nothing was
measured on the phone.

| | laptop, measured | phone, estimated |
|---|---|---|
| per box: crop + run + decode (`build/pose_test --bench`, median of 1001) | 1.36 ms (24 x 18 px box), 1.40 ms (70 x 55), 1.43 ms (220 x 160); the crop is 0.2-0.35 ms of it | 4-5 ms on a big core, 10-15 ms on a little one |
| against the learnt appearance head, the two runs interleaved under the same load | 1.30 ms against 0.83 ms: 1.56 x (16.0 M against 12.1 M multiply-accumulates, plus the larger input) | the appearance head was estimated at 3 ms a box on a big core: 4.7 ms |
| per pass at the unit's schedule, everything on, cap 4 (two rides, 5 minutes each) | passes with a pose: median 4.0 ms, 90 % point 6.4 ms (day) / 11.2 ms (night), 2.3 / 1.8 poses a pass; over all passes 2.0-2.1 ms a pass | a pass at the cap: 16-20 ms on a big core (worst case, 5 passes a second: a tenth of a big core); average about 7 ms a pass (3-4 % of a big core) |
| message | about 300 bytes per `pose`; the mean message of the day ride grows from 1889 to 2285 bytes (+21 %), of the night ride from 674 to 873 | |

**The cap.** At most `--lights-ru-pose-max` (4) runs a pass. On the two rides at the unit's schedule 220 of 728 and
40 of 457 passes with a pose were at the cap. With the whole frame and a cap of 2 on a busy city ride (13 vehicle
boxes a pass with the low list) the turn-taking gave the boxes of a chain a pose every 5 passes in the median (9 at
the 90 % point, 18 at worst); the order of the list changes from pass to pass, so a box can be passed over more
than once (one chain of eight boxes got none). No GPU work, no new thread, no new dependency; shed with the road
users.

**Why it matters on the unit**: in the unit's views 97 % of the first-list vehicle boxes of the day ride and 41 %
of the night ride's are cut at the bottom by the view (`clip` b): boxes that carry a bearing only today and for
which the head's `gn` is the only ground contact (section 4).

## 6. Not done, not verified

* **Nothing ran on the phone or the car.** The cost on the phone is an estimate from the project's documented
  factors; the head has never seen the car's own camera. signd itself (the daemon with the ring) was built with
  the option and not run live; everything went through `lights_replay`, which shares the pipeline code.
* **One training camera.** KITTI is one car in one town in daylight (2011, a roof camera 1.65 m up, focal length
  721 px at 1242 px width). The checks on other cameras (section 4) have no labels: they show where the head
  agrees with independent cues, not an accuracy. Night is the weak spot and is stated as such.
* **No wheel contact points.** KITTI has no wheel labels; the ground points are those of the 3D box's footprint.
  A data set with wheel keypoints (ApolloCar3D) needs a registration and was not used.
* **Persons and two-wheelers**: no pose (the head's heading for them is not usable). `--lights-ru-pose-all` exists
  for experiments only.
* **Vans and lorries**: the head's dimensions are a car's; its height for a van is better than the class prior of
  a `car` box and still low; for lorries it is wrong. The heading and the ground points of lorries and buses rest on
  few training examples (KITTI: 1 100 lorries, 500 trams, no buses).
* **The camera numbers** (`--lights-ru-pose-cam`) did not help on KITTI and could not be checked on another
  camera with labels; the option is there, unset.
* **One backbone with the appearance vector** (one crop, one run for both heads) was considered and not built:
  the appearance head works on 32 x 32 crops with a 5 % margin and is trained on ride chains, this one on 64 x 64
  with a 30 % margin on KITTI; a joint model needs the ride crops re-extracted and a joint training, and its
  weights would then derive from both sources. With both options on, a box costs the two runs.
* **The pass cap's turn-taking** has no unit test (it lives in `LightPipeline`, which needs the detector model);
  checked on a ride (section 5).
* **The consumer's half is design** (chris-roadsense `docs/lanes/detin-consumption-design.md`, section 9); nothing in
  chris-integrator reads `pose`.

## Reproduce

```sh
cmake --build build && ctest --test-dir build                      # pose: crop, box numbers, decode, JSON (stand-in model)
build/pose_test --bench pose1-m64.onnx                             # time per box, one thread
# a ride, the whole frame, every extracted frame (5 a second)
lights_replay --backend ort --model bdd100k_yolo11n_384x640.onnx --in 640x384 --roi 0,0,1,1 --no-track --road-users \
  --name-fps 20 --ru-scale --ru-flags --ru-view --ru-low 0.10 --ru-pose pose1-m64.onnx --ru-pose-max 99 FRAMES > out.jsonl
tools/pose/pose_check.py out.jsonl [--f F --cv HORIZON_ROW --cam-h H]
# against labels: chris-pose scripts/check_runtime.py (the same replay over KITTI pictures)
```
