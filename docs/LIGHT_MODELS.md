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
are COCO-class, not evaluated after the COCO YOLO11n result). A BDD-trained YOLOv8n (pure convolutions,
fully on Vulkan) exists (`dronefreak/bdd100k-yolov8n`, card mAP50 64.5 for lights) but was not evaluated
in L7.

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
