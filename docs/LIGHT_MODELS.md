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
