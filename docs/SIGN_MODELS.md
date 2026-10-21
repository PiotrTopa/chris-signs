# Traffic-sign model for the SD845 / ONNX Runtime CPU pipeline

Date: 2026-10-21. This is the model survey made for roadsense (moved here from
roadsense commit 246e68a). The survey ran in a scratch workspace that is not part
of this repository (`dl/` = downloads, `exp/` = all exports, `models/` = final
picks, `scripts/` = code, `out/` = annotated images + JSON results, `samples/` =
test photos); paths below of that form refer to it. The parts needed to
reproduce the chosen models are in this repository: `scripts/fetch-models.sh`,
`tools/scan_pt.py`, `tools/export_onnx.py`.

**Measured on the Poco F1 (after this survey):** signd's detector takes 114 ms
cold and 145-180 ms sustained on the four A75 cores, ~320 ms on the four A55
cores; live in the full stack (signd on the little cores, modeld on the big
ones) it delivers 1.5-2 frames/s. The estimates in section 3 below were
optimistic.

## TL;DR

* **Chosen: a two-stage Mapillary (MTSD) pipeline** from `jkot/pova-traffic-sign-recognition-models` (Brno University of Technology FIT student project "PoVa"; authors V. Tlustoš, J. Vlasák, J. Kotoun):
  1. `models/mtsd_signdet_yolov8n_384x640.onnx` — YOLOv8n **class-agnostic sign detector** (1 class `traffic-sign`), trained on MTSD at 1280 px, exported here at a static 384x640.
  2. `models/mtsd_signcls_yolov8n_64.onnx` — YOLOv8n-cls **400-class MTSD sign classifier** on 64x64 crops (`models/mtsd_signcls_classes.txt`).
  It was the only candidate to get every sign in the Polish test photos right, including Polish yellow warning signs (see Results).
  Laptop latency: detector **22.5 ms** (4 threads), classifier **0.53 ms per crop**. My estimate for the SD845 (4x A75) is about 80–120 ms per frame, roughly 8–12 fps (not measured on the device).
* **License caveat (important):** the HF repo and its GitHub source (`Jkotoun/pova-traffic-sign-recognition`) have **no license at all**, so by default all rights are reserved. The weights are also trained with **Ultralytics** (AGPL-3.0 claim on trained weights) on **MTSD** (Mapillary dataset terms: CC BY-NC-SA, research / non-commercial). That is fine for a private, non-commercial car project, but **do not redistribute the weights or the ONNX**. `scripts/fetch-models.sh` downloads the `.pt` from HF at a pinned commit and exports it locally (recipe below); for anything beyond that, ask the authors for a license.
* **Secondary (general road objects):** `models/bdd100k_yolo11n_384x640.onnx` (`dronefreak/bdd100k-yolo11n`): 10 BDD100K classes `person, rider, car, truck, bus, train, motor, bike, traffic light, traffic sign`. Laptop latency 24 ms. **AGPL-3.0**. Its "traffic sign" boxes can also be fed into the MTSD classifier.
* There is **no permissive (Apache/MIT/BSD) per-class sign detector that is fast enough**. The permissive candidates are all too slow or too coarse: Signfeld RetinaNet-R50 (MIT), aarcosg TF1 SSD (MIT, 3 super-classes), RF-DETR-Nano on BDD100K (Apache-2.0, 234 ms on the laptop, so about 1 s on the SD845). All usable per-class sign models found are Ultralytics-trained.

## 1. Candidate survey

Latency = ONNX Runtime 1.31 CPU EP on this laptop (i7-8850H, 6C/12T, AVX2), static 1x3x384x640 input, median of 30 runs, `intra_op_num_threads` = 4 / 1, `ORT_ENABLE_ALL`. Quality figures are the authors' own validation numbers unless marked "ours".

| # | Candidate (source) | Weights license (as stated) + training stack | Classes | Arch / params / GFLOPs @384x640 | Train size | ONNX | Quality evidence | Laptop ms (4t / 1t) | Verdict on PL photos (ours) |
|---|---|---|---|---|---|---|---|---|---|
| **A** | **jkot/pova-traffic-sign-recognition-models** (HF, commit `0c13500`), det `binary_detector-nano-1280.pt` + cls `yolo_classifier_nano.pt` | **none stated**; Ultralytics 8.0.x (AGPL claim); MTSD data (CC BY-NC-SA) | det: 1 (`traffic-sign`); cls: **400 MTSD classes** (regulatory / warning / information / complementary, per sign type) | YOLOv8n 3.0 M / 4.9 GF + YOLOv8n-cls 1.95 M / ~0.3 GF per 64x64 crop | 1280 / 64 | exported here (opset 17) | Binary det MTSD-val mAP50 70.5 % @1280 (57 % for the 640-trained n); cls top-1 89.3 % / top-5 96.7 % (MTSD val) | **22.5 / 62.7** + 0.53 per crop | **every near/mid sign detected and classified correctly** (11 signs, incl. yellow PL warnings); 1 low-conf FP (0.27); misses far (<15 px) signs |
| B | AlessandroFerrante/StreetSignSenseY12n | CC-BY-4.0 (card), Ultralytics 8.3 (AGPL claim) | 63 (EU-style: prio/forb/warn/mand/info) | YOLO12n 2.6 M / 4.3 GF | 640 | exported here | mAP50 86.9 / mAP50-95 76.1 (own val) | 22.5 / 58.8 | speed signs right; missed no-right-turn, children, both yellow warnings; no-parking at 0.43 |
| C | OrDinaD/belarus-road-signs-yolov8 | MIT (card); Ultralytics 8.4 (AGPL claim) | 249 (Belarus, Vienna-convention) | YOLOv8n 3.4 M | **320** | yes (640x640 opset 12); re-exported 384x640 | mAP50 72.5 (own val, 20 epochs) | 28.9 / 74.0 | poor: 50 read as 80, missed several, odd FPs |
| D | JakobJFL/yolov8-dk-Traffic-Signs | Apache-2.0 (card); Ultralytics 8.0 (AGPL claim) | 19 (Danish speed limits, ends, no-parking/stopping only) | YOLOv8s 11.1 M / ~17 GF | 1024 | exported here | mAP50 ≈ 0.96 (results.csv) | 63.6 / 196 | correct on its few classes; too narrow + 3x slower |
| E | Panoramax/classify_de_road_signs (+ detect_face_plate_sign) | etalab-2.0 (permissive, CC-BY-like); Ultralytics (AGPL claim) | 134 German sign types (classifier); detector is generic sign/plate/face | YOLO26l-cls 13 M @224 (heavy per crop) + YOLOv8s @2048 | 224 / 2048 | cls exported (`exp/pnx_de_cls_224.onnx`) | top-1 98.7 % (own val) | not benchmarked end-to-end | not evaluated: classifier ~50x heavier than A's, detector trained for 2048 px panoramas |
| F | moabitcoin/Signfeld `DE-signs-169.pth` (GitHub release) | **MIT**; detectron2 | 169 German sign types | RetinaNet R50-FPN, 318 MB, ~100 GF | — | needs detectron2 export | GTSDB mAP 67.2 (trained on synthetic data only) | not run (≥10x over budget) | rejected: far too heavy for the SD845 |
| G | aarcosg/traffic-sign-detection (Google Drive) | **MIT**; TF1 Object Detection API | 3 GTSDB super-classes (prohibitory/danger/mandatory) | SSD-MobileNet-v1 5.6 M / 2.3 GF@300 | 300 | TF1 frozen graph, needs tf2onnx + TF NMS removal | GTSDB mAP 61.6 (small signs 22 %) | not run | rejected: coarse classes, weak on small signs, 2018 TF1 tooling |
| H | dronefreak/bdd100k-rfdetr-nano | **Apache-2.0**; rfdetr | 10 BDD100K classes (`traffic sign` = one class) | RF-DETR-Nano (DINOv2-S windowed) 30.5 M | 576 | exported here (`exp/rfdetr*/`, opset 17, uses GridSample/TopK) | mAP50 56.9 / mAP50-95 31.6 (BDD test) | **292** (576²) / 234 (384x640) | good boxes, but ~10x too slow for the SD845 |
| **I** | **dronefreak/bdd100k-yolo11n** (secondary pick) | **AGPL-3.0** (card); Ultralytics 8.4 | 10 BDD100K classes | YOLO11n 2.6 M / 3.9 GF | 960 | exported here | mAP50 51.6 / mAP50-95 29.1 (BDD test) | **24.0 / 60.9** | finds cars, persons, all larger signs (class-agnostic) |
| — | Other HF repos (cvtechniques LISA = US signs; ankitjha07 / yahyagul / AI4Ev3r / Nughm3 = unknown or non-EU sets; neitpm/star092304 = Vietnamese; Genius-Society TT100K = Chinese; sergiopaniego / maier-s DETR-R50 = ~86 GF; merve RF-DETR roadsign = mAP 0.19, Asian set) | mixed | — | — | — | — | — | — | not suitable (wrong sign family, too heavy or too weak) |
| — | ONNX Model Zoo / OpenVINO OMZ | Apache-2.0 / MIT | COCO only (`stop sign`, `traffic light`) | — | — | — | — | — | no traffic-sign model exists |
| — | Mapillary MTSD paper baselines, PUT Poznań YOLOv4-MTSD (PP-RAI 2022) | — | 300–400 | Faster R-CNN / YOLOv4 @960 | — | — | AP50 59 % (PUT) | — | weights not released |

## 2. Chosen pipeline: MTSD two-stage (A)

### Files (final, under `models/`)

| file | sha256 | input | output |
|---|---|---|---|
| `mtsd_signdet_yolov8n_384x640.onnx` (12.2 MB) | `6928bdc9b054f241db4e48831c6db6f7a14222b722f7ebaa9693fe01342ee51c` | `images` float32 **[1,3,384,640]** | `output0` float32 **[1,5,5040]** |
| `mtsd_signcls_yolov8n_64.onnx` (7.8 MB) | `01dafa0e2db29cf39993a0545150e24ffcd552ccc29827221678de114b1ba986` | `images` float32 **[1,3,64,64]** | `output0` float32 **[1,400]** (softmax probabilities) |
| `mtsd_signdet_classes.txt` | `1adfbf92…` | 1 line: `traffic-sign` | |
| `mtsd_signcls_classes.txt` | `55d766bd…` | 400 lines, line i = class id i | |

All exported files are ONNX opset 17, IR version 8, static shapes, batch 1, no custom ops and no external data. The detector uses only these ops: Conv, Sigmoid, Mul, Add, Split, Concat, MaxPool, Resize (nearest), Reshape, Transpose, Softmax, Slice, Sub, Div. The classifier uses Conv, Sigmoid, Mul, Add, Split, Concat, GlobalAveragePool, Flatten, Gemm, Softmax. Class names are also embedded in the ONNX metadata key `names`.

### Source downloads (pinned in `scripts/fetch-models.sh`)

| file | URL | sha256 |
|---|---|---|
| detector `.pt` (24.7 MB) | `https://huggingface.co/jkot/pova-traffic-sign-recognition-models/resolve/0c135006544630d813603f48866de2f11287c518/binary_detector-nano-1280.pt` | `cf5fe8c5c2c3cb1ff77b8232778c4f95d8321ddeb2d9bad7928e03e3373db444` |
| classifier `.pt` (23.6 MB) | `https://huggingface.co/jkot/pova-traffic-sign-recognition-models/resolve/0c135006544630d813603f48866de2f11287c518/yolo_classifier_nano.pt` | `9124dd54de24d4b0b22eb7d6524544b74cbda24c4258f75441c53a67a7a6fef4` |

The `.pt` files are pickles. `tools/scan_pt.py` lists every global they import, and both import only `torch.*`, `ultralytics.*`, `collections.*`, `dill._dill._load_type` and builtins. Run that scan before loading the files. The other variants in the repo (`binary_detector-small-1280.pt`, `-medium-1920.pt`, `yolo-classifier-medium.pt`) are more accurate (det mAP50 73.9 / 77.3, cls 93.1 %) but cost 3x to 10x more compute.

### Export recipe (dev machine, not the car)

`scripts/fetch-models.sh` automates all of this (download with sha256 check,
pickle scan, venv `.venv-export`, export, I/O-shape check); by hand:

```
python3 -m venv venv && venv/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
venv/bin/pip install ultralytics==8.4.183 onnx onnxslim dill      # the versions tested
venv/bin/python -I tools/export_onnx.py binary_detector-nano-1280.pt models/mtsd_signdet_yolov8n_384x640.onnx 384 640 models/mtsd_signdet_classes.txt
venv/bin/python -I tools/export_onnx.py yolo_classifier_nano.pt     models/mtsd_signcls_yolov8n_64.onnx 64 64 models/mtsd_signcls_classes.txt
```
(`export_onnx.py` writes the class list from the model's `names` and calls `YOLO(pt).export(format='onnx', imgsz=(H,W), opset=17, simplify=True, dynamic=False, nms=False, half=False, batch=1)`.) Re-exports may not be byte-identical across library versions, so verify a re-export by its I/O shapes and outputs, not by sha256. Any other static size that is a multiple of 32 also works (e.g. 448x768 or 512x896 for better small-sign recall; the cost scales with pixel count).

### Preprocessing — detector

1. Take the camera frame as **RGB** uint8, size W0xH0. If it comes from OpenCV it is BGR and must be swapped.
2. **Letterbox** to 640x384 (W x H): `r = min(640/W0, 384/H0)`, `nw = round(W0*r)`, `nh = round(H0*r)`. Resize bilinearly to nw x nh. `dw = (640-nw)/2`, `dh = (384-nh)/2`, `left = round(dw-0.1)`, `top = round(dh-0.1)`. Paste onto a 640x384 canvas filled with (114,114,114). If the camera already delivers 640x384, then r = 1 and there is no padding.
3. Convert to float32, **divide by 255** (no mean/std), lay out as **NCHW** in RGB plane order, shape [1,3,384,640], contiguous.

### Postprocessing — detector (`output0` [1, 4+nc, 5040], nc = 1 here)

* 5040 anchors = 80x48 (stride 8) + 40x24 (stride 16) + 20x12 (stride 32). Layout is channel-major: `out[c*5040 + i]`.
* Rows 0..3 = **cx, cy, w, h in letterboxed-input pixels**, already decoded (DFL and stride applied inside the graph). Rows 4..4+nc-1 = per-class scores, **already sigmoided**. There is no separate objectness score.
* For each anchor i: `score = max_c out[4+c][i]`, `cls = argmax`. Keep the anchor if `score > conf_thr`. Box = `(cx-w/2, cy-h/2, cx+w/2, cy+h/2)`.
* **NMS** is greedy IoU NMS, sorted by score, IoU threshold 0.45, per class (with nc = 1 that is just plain NMS). Keep at most 100 boxes.
* Map back to the frame: `x = (x - left) / r`, `y = (y - top) / r`, then clip to [0,W0] x [0,H0].
* Suggested `conf_thr`: 0.25 for recall, **0.35–0.4 in the car** (the only false positive in the tests scored 0.27).

### Preprocessing — classifier (per detected box)

1. Crop the box from the **original RGB frame** (I used no padding; the sign fills the crop).
2. Do the Ultralytics classify val transform: resize so the **shorter side = 64**, keeping aspect ratio (bilinear), then **center-crop 64x64**. For near-square sign boxes a plain stretch to 64x64 is practically equivalent.
3. Convert to float32, **/255**, no mean/std, RGB, NCHW [1,3,64,64]. Batch-1 calls in a loop are fine (~0.5 ms each).

### Postprocessing — classifier

`output0` [1,400] already holds softmax probabilities. Take `k = argmax` and `p = prob[k]`; `label = mtsd_signcls_classes.txt[k]`. Names follow the pattern `<group>--<sign-name>--g<variant>`, e.g. `regulatory--maximum-speed-limit-40--g1`. Strip `--g\d+` to merge graphical variants. Suggested: accept when p ≥ 0.6, and smooth over 3–5 frames (track IoU + majority vote) before acting on a sign.

### Reference implementation

`scripts/run.py` (survey workspace, not in this repository) is a numpy + PIL implementation of exactly this spec. I verified its boxes against `ultralytics.YOLO(onnx).predict` (same detections, coordinates within 1–3 px; the remaining difference comes from PIL vs cv2 bilinear resize). Usage:
```
venv/bin/python -I scripts/run.py --det models/mtsd_signdet_yolov8n_384x640.onnx --det-names models/mtsd_signdet_classes.txt \
  --cls models/mtsd_signcls_yolov8n_64.onnx --cls-names models/mtsd_signcls_classes.txt --out out --tag pova samples/*.jpg
```

### ORT C++ session settings

`SetIntraOpNumThreads(4)`, `SetInterOpNumThreads(1)`, `ORT_SEQUENTIAL`, `GraphOptimizationLevel::ORT_ENABLE_ALL`, CPU EP. On the SD845, pin the process to the A75 cluster (cpus 4-7, e.g. `taskset -c 4-7`). A55 cores would only slow the 4-thread pool down. Create the two sessions once and reuse the input buffers.

## 3. Results on real Polish street photos (Wikimedia Commons)

Photos (1920 px Commons thumbnails) are in `samples/`:
* `poznan_daszewicka.jpg` — "Daszewicka Street in Poznań", MOs810, CC BY 4.0
* `trzebnica.jpg` — "Trzebnica, Poland - panoramio (27)", bahryn, CC BY 3.0
* `samokleski.jpg` — "Samoklęski, Lublin Voivodeship, Poland, June 2001", Kgbo, CC BY-SA 4.0
* `poznan_hlonda.jpg` — "Hlonda Street Poznań", Rzuwig, CC BY 3.0 (wide overpass view, small signs)
* extra: `gradowskiego.jpg` ("Ulica Gradowskiego", CC BY-SA 4.0), `burowiecka.jpg` ("Katowice Burowiecka", CC BY-SA 4.0)

Chosen pipeline A. The format is det conf / cls prob. Annotated images are `out/pova_*.jpg`, and `out/contact_pova.jpg` is a 2x2 contact sheet.

| image | detections | assessment |
|---|---|---|
| poznan_daszewicka | `maximum-speed-limit-40` 0.90/1.00, `no-parking` 0.89/1.00, `warning--double-curve-first-right` 0.76/1.00 | all correct (PL B-33 40, B-35, yellow A-3 type); the T-arrow plate and the sign backs were correctly ignored |
| trzebnica | `no-right-turn` 0.78/1.00, `maximum-speed-limit-40` 0.77/0.99 | both correct (B-22, distant 40) |
| samokleski | `maximum-speed-limit-50` 0.91/1.00, `warning--children` 0.84/1.00 | both correct (yellow PL A-17); the town-name sign was correctly not called a sign |
| poznan_hlonda | `keep-right` 0.63/1.00, `yield` 0.27/0.93 | keep-right correct (C-9 on the island); `yield` is a **false positive** on a red/white U-3 delineator. **Misses**: several far-away signs (~10–15 px), as expected at 640 px |
| gradowskiego | `information--parking` 0.86/0.92, `one-way-straight` 0.64/1.00 (tiny blue sign far left) | parking correct; street-name boards ignored |
| burowiecka | none | correct (only a street-name board) |

Comparison images for the rejected models are `out/sss_*.jpg` (B), `out/by_*.jpg` (C), `out/dk_*.jpg` (D), `out/bddy11n_*.jpg` (I) and `out/rfdetr576_poznan_hlonda.jpg` / `out/rfdetr384x640_poznan_hlonda.jpg` (H). Raw JSON is in `out/*.json`.

### Latency (this laptop, i7-8850H, ORT 1.31 CPU EP, median / p90 ms)

| model | 4 threads | 1 thread |
|---|---|---|
| MTSD detector 384x640 | 22.5 / 22.8 | 62.7 / 67.1 |
| MTSD classifier 64x64 (per crop) | 0.53 / 0.55 | 0.66 / 0.73 |
| BDD100K YOLO11n 384x640 (secondary) | 24.0 / 25.2 | 60.9 / 61.6 |
| StreetSignSense YOLO12n 384x640 | 22.5 / 23.2 | 58.8 / 59.3 |
| RF-DETR-Nano BDD 576x576 / 384x640 | 292 / 234 | 816 / 630 |

**SD845 estimate (not measured):** an A75 at 2.8 GHz with NEON fp32 runs roughly 2.5–4x slower than an AVX2 Coffee Lake core under ORT. That puts the detector at **~60–100 ms with 4 big cores** and the full pipeline (with ~5 signs per frame) at about 65–105 ms, i.e. **~10–15 fps**. That leaves enough margin over 5 fps that both the detector and the BDD YOLO11n could run (~130–200 ms combined) if needed. Confirm with `onnxruntime_perf_test -x 4 -e cpu -r 100 <model>` on the device. If it falls short, INT8 dynamic or static quantization through ORT (QDQ, ARM64 kernels) usually gives 1.5–2.5x.

## 4. Secondary: general road-object detector

`models/bdd100k_yolo11n_384x640.onnx`, sha256 `19a6471318c0b5dad61010d53154acccbe632581891575d78e8914df2d9be2e1`. Input `images` [1,3,384,640], output `output0` [1,14,5040]. Classes are in `models/bdd100k_classes.txt` (10 lines, id order: person, rider, car, truck, bus, train, motor, bike, traffic light, traffic sign).
* Source: `https://huggingface.co/dronefreak/bdd100k-yolo11n/resolve/2b5554133522f96f50ce87a4df0a2de556e4b7ad/best.pt`, sha256 `c120d075bcc6aaf94523f9a2c9d681819ebd692f36eb5ea64656d1c5f1278a70`. Trained at 960 px; BDD100K test mAP50 51.6 / mAP50-95 29.1. **License AGPL-3.0** (stated on the card).
* Pre- and postprocessing are **identical** to the MTSD detector (letterbox, /255, RGB NCHW; output rows cx,cy,w,h + 10 sigmoid scores). Use class-aware NMS (e.g. add `cls*4096` to the coordinates before NMS).
* In tests it found all larger signs (class-agnostic) plus cars and pedestrians. Its `traffic sign` boxes can be passed to the MTSD 400-class classifier.
* The permissive alternative is `dronefreak/bdd100k-rfdetr-nano` (Apache-2.0). It is exported and works (`exp/rfdetr/rfdetr-nano.onnx`; input `input` [1,3,576,576]; resize **without** letterbox; RGB /255 then ImageNet mean (0.485,0.456,0.406) / std (0.229,0.224,0.225); outputs `dets` [1,300,4] = cx,cy,w,h normalised to 0..1, and `labels` [1,300,11] = logits → sigmoid; class id = column 0..9, column 10 unused; take the top-300 over (query, class) with no NMS). At **234–292 ms on the laptop it is too slow for the SD845** at 5 fps. Reference implementation: `scripts/run_rfdetr.py`.

## 5. Caveats

* **License / redistribution:** chosen model A has no license file (all rights reserved by default), Ultralytics claims AGPL-3.0 over trained weights, and MTSD is CC BY-NC-SA. Keep it as a user-side download and export; do not commit or re-host the weights or ONNX; ask the authors (Josef Kotoun, HF `jkot`) for an explicit license if this ever goes beyond personal use. All Ultralytics-trained alternatives (B, C, D, E, I) carry the same AGPL claim regardless of the license on their model cards.
* **Small signs:** the detector was trained at 1280 px, and at 640x384 signs below about 15 px are missed (Hlonda photo). Options: run at 448x768 or 512x896 (cost ~1.4x / ~1.9x), crop a right-hand/upper ROI from a higher-resolution frame, or rely on tracking, since signs grow as the car approaches.
* **Closed set:** the classifier always outputs one of 400 MTSD classes and has no "other/unknown" class. Gate on classifier probability plus multi-frame agreement. MTSD covers Polish signs via the `--gN` variants (yellow warning signs were classified correctly), but Polish-specific signs such as speed-limit zone signs and the various D/E/T plates may map to the nearest MTSD class.
* **False positives:** red/white delineators and similar shapes can trigger the detector at low confidence. Use det conf ≥ 0.35.
* **Evaluation size:** the qualitative check covers 6 photos (11 near/mid-range signs) and is not a benchmark. The quantitative figures are the authors' MTSD validation numbers.
* **Latency on the SD845** was extrapolated here; the device measurements are at the top of this file.
