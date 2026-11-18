# L24-S: which steps improve the sign and traffic-light outputs of signd

Sub-lane L24-S of roadsense lane L24 (2.0 campaign), 2026-11-18 21:30 - 2026-11-19 ~01:00 CET, laptop only
(the phone was not touched; it was not reachable after the laptop restart at 22:28 CET). Branch `v2-l24s` of
roadsense-signs, from `v2` (b50c08a). Owner's ruling: quality first, then the same quality for less; no
regression in measurement quality.

## Summary

| step | quality on the labelled rides | cost on the phone | verdict |
|---|---|---|---|
| 1. sign detector 5 -> 7.5 -> 10 runs/s | reference signs listed 0.74 -> 0.84 (10 Hz, pooled over three rides), first listing 0.2-0.3 s earlier; human labels: within the phase-to-phase noise (false rules 23-25 -> 26, missed 36-40 -> 35) | +51 ms of GPU per extra run: 10 Hz takes about 5 road runs/s away (13.8 -> ~8.5) | **reject now** (needs GPU room, e.g. the v0.11.1 road model) |
| 2. second full-scale sign view on alternate runs (window x 380-1020, y 192-576) | signs under 24 px: 12-16 px sign-time 0.06 -> 0.23, 16-24 px 0.34 -> 0.45; **but** 24+ px 0.79 -> 0.62 and reference signs listed 0.74 -> 0.71 (warsaw-city, narrow camera: 389 -> 350); human labels: missed 38 -> 39.5, matched 58.5 -> 56.5 | none on the GPU (same run count); CPU about the same | **reject** (regression on large signs); option kept for experiments |
| 3. the light detector's `traffic sign` class as sign proposals | reference signs listed 0.74 -> 0.81 (`--lights-signs`), first listing +0.4 s; human labels: warnings found 36 -> 39-41, momentary 109 -> 111-113, false rules 23 -> 23-25 (noise ±2), more engine events (momentary 211 -> 219-233) | no detector run; ~1.5 classifier crops per light frame (7-8 crops/s on the big cores) | **A/B on the phone** (`--lights-signs-keep --lights-signs-nohits` = no change at the engine within noise, +5 warnings; `--lights-signs` = most recall, more engine output); not a default |
| 4. light detector | see section 4 | | see section 4 |
| 5. arrows as a light state | no arrow labels exist (0 of the 320 labelled car crops are marked as arrows; BDD100K has no arrow attribute) | | **needs data** (section 5) |

## 0. Method

### Labelled rides and references

* Signs: the six L11 rides with human labels (`krakow-2024`, `rural-zloczew`, `forest-vistula-spit`,
  `s1-roadworks`, `warsaw-city`, `katowice-rain`; roadsense `docs/lanes/L11-signs-in-effect/labels/`), the L11
  phone recordings (`~/l11/rec/<ride>.jsonl`: every camera frame, the road and lights messages), the ride
  videos (`~/datasets/rides/<ride>/*.mjpeg`). Models as installed on the phone: detector
  `mtsd_signdet_yolov8n_384x640`, L12 classifier (402 classes), L8 verifier, `--class-min-score
  regulatory--no-entry=0.71 --dist-match-new 2.0`.
* Two scores per sign variant:
  1. **Human labels (decisive for the rules).** The variant's `signs` messages replace the recorded ones in the
     L11 recording (after the camera message of the processed frame), the api's signctx annotates them, the
     roadsense rules engine (`tools/rules_replay.py`, branch v2) turns them into rules, L11 `score.py` scores
     them: false rules, missed / matched intervals of lasting rules, momentary signs and warnings found (+-6 s),
     interval start delay. The engine's thresholds (hits) are tuned for 5 Hz and were not changed.
  2. **"Detector at its best" reference (by size).** No box labels exist, so as in L15 for the lights the
     reference is the same detector with everything it has: the whole frame plus three 640x384 tiles at
     scale 1.0 (the detector's training scale; the whole frame is seen at 0.5) on the horizon band (y
     168-552), every second frame (10 Hz), linked by signd's own tracker; a reference sign is a track of >= 4
     hits whose class is a sign (not `background`, not suppressed). Three rides (krakow-2024, rural-zloczew,
     warsaw-city; 942 reference signs, 38,100 sign-frames). Numbers are shares of what the detector can see
     at full scale: upper bounds of true rates. Reported: share of the sign-time at which the newest message
     (<= 0.35 s old) lists a non-suppressed track on the sign, by box size (max side); reference signs
     listed at least once; median time from the first listing to the sign's last sighting (~ passing),
     paired with the base run.
* Before any of this, where small signs sit (1 frame/s probe of three rides, whole frame + six full-scale
  tiles): reference signs of 12-16 px are found by today's whole-frame view in 3-30 % of their frames, 16-24 px
  in 35-57 %, 24+ px in 85-92 %. They sit on the horizon band (y 330-440 of 720), spread over x 250-1190
  (median x 640-860). A centred 640x384 window (x 380-1020, y 192-576) holds 90-100 % of the 12-16 px ones and
  60-90 % of the 16-24 px ones.

### Tools (all in this repository)

* `tools/signs_eval/cache_dets.py`: per-frame detections of a ride for any set of views (whole frame, window,
  tiles, the light detector's views), exactly as `SignPipeline::process` (signd's own NV12 conversion and
  resampler through `libsigns_pre.so`, same letterbox, decode, NMS, mapping, classifier crop, glow and
  verifier), with the ONNX graphs run on the laptop GPU by `onnx_torch.py`. Checked: 131 detections on 50
  krakow frames against `signd_replay --raw`: same boxes (max 0.05 px), scores (max 5e-4), classes (all).
* `tools/signs_track_replay.cc`: signd's tracker (same code) on those detections for a schedule of processed
  frames (rate, phase, view pattern, light-thread updates). Checked: identical `signs` lines to
  `signd_replay` on the same 50 frames.
* `tools/signs_eval/variant.py` (schedule + run), `score_ref.py` (reference scores), `ride_rules.py` +
  `eval_variant.sh` (human-label scores), `compare_rules.py` (tables), `merge_parts.py` / `merge_ride.sh`.
* `tools/lights/replay_score.py` (L15) with `lights_replay` for the lights.

Reproduce (laptop; models and scratch paths as in the lane): see the commands under each table. Detections:
`~/datasets/l24s/run_cache.sh` (rides in parts of 2000 frames; krakow-2024 every frame with the tiles, the other
rides every second frame), then `tools/signs_eval/merge_ride.sh <ride>`. Light-thread views were cached on every
4th frame.

## 1. Sign detector rate

Reference (three rides, pooled; `score_ref.py <ride> ref/<ride>.jsonl ...` on the outputs of
`variant.py <ride> OUT --hz H [--phase P]`):

| variant | sign-time listed, all | 12-16 px | 16-24 px | 24+ px | reference signs listed (krakow / rural / warsaw) | pooled | first listing to passing, median s |
|---|---|---|---|---|---|---|---|
| 5 Hz (today), phase 0 | 0.572 | 0.058 | 0.341 | 0.789 | 216/282, 99/144, 389/516 | 0.747 | 1.8 / 0.9 / 1.7 |
| 5 Hz, phase 2 | 0.571 | 0.057 | 0.342 | 0.787 | 216/282, 98/144, 382/516 | 0.739 | 1.6 / 0.8 / 1.7 |
| 7.5 Hz (krakow only, needs every frame) | krakow 0.509 vs 0.477 | 0.098 vs 0.089 | 0.455 vs 0.410 | 0.698 vs 0.665 | 227/282 | | 1.8 |
| 10 Hz | 0.634 | 0.079 | 0.425 | 0.847 | 233/282, 134/144, 425/516 | 0.841 | 1.95 / 1.1 / 2.0 |
| 10 Hz, max-age 20 (same time as 10 at 5 Hz) | krakow 0.533 | | | | 233/282, 134/144, 425/516 | | same as 10 Hz |

No sign under 12 px is listed by any whole-frame variant (the detector does not see them at scale 0.5). The
rural ride (higher speed) gains most: at 5 Hz a third of its reference signs are never listed.

Human labels (six rides; `eval_variant.sh NAME PHASES "--hz H"`, `compare_rules.py ~/datasets/l24s/eval ...`):

| per pass of the six rides | 5 Hz phase 0 | 5 Hz phase 2 | 10 Hz |
|---|---|---|---|
| false rules | 25 | 23 | 26 |
| missed intervals | 40 | 36 | 35 |
| matched intervals | 58 | 59 | 60 |
| momentary found / engine events / human | 108 / 207 / 209 | 109 / 211 / 209 | 110 / 246 / 209 |
| warnings found / engine events / human | 37 / 90 / 61 | 36 / 83 / 61 | 41 / 99 / 61 |
| interval start delay, median s | -0.4 | -0.5 | -1.35 |

The two phases of the same 5 Hz pipeline differ by 2 false rules and 4 missed intervals: that is the noise
of these six rides. 10 Hz finds 4-5 more warnings and starts lasting rules ~0.9 s earlier, and the engine
emits 35 more momentary events (unlabelled or false; its hit thresholds are 5 Hz ones).

Cost (phone, from L21: detector 51 ms of GPU per run, the sign thread 21.6 ms of big-core CPU per frame):
7.5 Hz = +128 ms GPU/s (+13 % of the GPU) and +54 ms CPU/s; 10 Hz = +255 ms GPU/s (+25 %) and +108 ms CPU/s.
The GPU is full in time (91 %), so every extra detector run costs about one road run (51 vs 47 ms): 10 Hz takes
the road model from ~13.8 to ~8.5 runs/s. **Verdict: reject for now**; the gain is real by the reference and is
the first thing to buy if the v0.11.1 road model frees 9-19 % of the GPU (L21 section 5).

## 2. A second, full-scale sign view

`signd --roi2 x0,y0,x1,y1 [--roi2-every N]`: every N-th processed frame (default 2) the detector looks at the
window instead of the whole frame, at the scale that fits it into the 640x384 input (window 640x384 px =
scale 1.0). Boxes cut by an inner window edge are dropped; the tracker does not age tracks outside the window
on those frames (they end after 3 x max_age frames without a match at the latest). Window tested: x 380-1020,
y 192-576 of 1280x720 (`--roi2 0.296875,0.2666667,0.796875,0.8`), chosen from where the small reference signs
sit (section 0).

| variant | sign-time listed, all | <12 px | 12-16 px | 16-24 px | 24+ px | reference signs listed (k / r / w) | pooled | first listing to passing, s |
|---|---|---|---|---|---|---|---|---|
| 5 Hz (today), phase 0 | 0.572 | 0.000 | 0.058 | 0.341 | 0.789 | 216, 99, 389 | 0.747 | 1.8 / 0.9 / 1.7 |
| 5 Hz, window on every 2nd | 0.521 | 0.080 | 0.229 | 0.448 | **0.615** | 216, 100, **350** | 0.707 | 2.4 / 1.0 / 2.2 |
| 5 Hz, window on every 3rd | 0.534 | 0.022 | 0.146 | 0.408 | 0.674 | 222, 98, 367 | 0.729 | 2.35 / 1.0 / 2.1 |
| 10 Hz, window on every 2nd (10 runs/s) | 0.582 | 0.138 | 0.324 | 0.528 | 0.660 | 251, 124, 404 | 0.827 | 2.65 / 1.2 / 2.4 |

Human labels (six rides):

| per pass of the six rides | 5 Hz (2 phases) | window every 2nd (2 phases) | window every 3rd | 10 Hz + window |
|---|---|---|---|---|
| false rules | 24 | 25.5 | 26 | 26 |
| missed intervals | 38 | 39.5 | 40 | 34 |
| matched intervals | 58.5 | 56.5 | 58 | 57 |
| momentary found / engine | 108.5 / 209 | 106.5 / 195.5 | 106 / 194 | 107 / 233 |
| warnings found / engine | 36.5 / 86.5 | 37.5 / 85.5 | 37 / 84 | 38 / 95 |

The window does what it is for (small signs listed 2-4 times as often, and seen ~0.5 s earlier), but the
large signs outside it are then looked at only 2.5 times a second and the tracker loses them: 24+ px sign-time
0.79 -> 0.62, and on warsaw-city (narrow camera, signs leave the frame early) 39 fewer reference signs listed.
The human-label scores do not move beyond the noise. **Verdict: reject** (a regression on large signs is not
acceptable); the option stays (off by default) for a camera whose geometry differs, or for a phone with GPU
room for 10 runs/s, where it is the best variant by reference recall. Cost: no extra detector run; laptop
CPU per frame: letterbox 10.8 ms (whole) vs 9.3 ms (window), classifier 5.5 vs 7.9 ms (1.19 vs 1.47 boxes
per run).

## 3. The light detector's `traffic sign` class

The light detector (BDD100K YOLO11n) computes a `traffic sign` score (class 9) in every light run, at scale
0.72 (wide view) and 1.0 (zoom view) where the sign detector sees 0.5. `signd --lights-signs`: the light
thread keeps those boxes (score > `--lights-signs-thr`, 0.35), classifies them with the sign classifier (+
glow, verifier), and hands them to the sign thread, which runs a tracker update of its own for that light
frame (`seen` = the light view, so tracks outside it are not aged). The boxes are marked external: they never
set a track's best score or crop. A light update older than the last published message publishes nothing
(the topic's seq stays monotonic) and its ended tracks go out with the next message; on the laptop
(slow ORT) 109 of ~120 light updates arrived late, so the replay variants below are scored that way
("quiet", the realistic case) unless marked "published". `--lights-signs-keep`: those boxes only continue
tracks the sign detector started. `--lights-signs-nohits`: they do not count as hits (the reported `hits`
keep their meaning; the boxes keep tracks alive, placed and voting).

Replay: sign frames at 5 Hz phase 2, light frames at 5 Hz on the frames in between, alternating wide / zoom
views as signd does (`variant.py <ride> OUT --hz 5 --phase 2 --lights LW,LZ -- [--quiet-light]
[--external-keep] [--external-nohits]`).

| variant | sign-time listed, all | 12-16 px | 16-24 px | 24+ px | reference signs listed (k / r / w) | pooled | first listing to passing, s |
|---|---|---|---|---|---|---|---|
| 5 Hz (today), phase 2 | 0.571 | 0.057 | 0.342 | 0.787 | 216, 98, 382 | 0.739 | 1.6 / 0.8 / 1.7 |
| + light proposals, published | 0.619 | 0.161 | 0.478 | 0.778 | 245, 133, 422 | 0.849 | 2.45 / 1.2 / 2.2 |
| + light proposals (`--lights-signs`) | 0.616 | 0.085 | 0.420 | 0.818 | 228, 130, 403 | 0.808 | 2.0 / 1.1 / 2.1 |
| + keep only (`--lights-signs-keep`) | 0.594 | 0.061 | 0.379 | 0.806 | 222, 128, 399 | 0.795 | 1.8 / 1.0 / 1.9 |
| + keep, no hits (`--lights-signs-keep --lights-signs-nohits`) | 0.568 | 0.047 | 0.332 | 0.789 | 219, 113, 390 | 0.766 | 1.55 / 0.8 / 1.7 |

Human labels (six rides, phase 2; `eval_variant.sh NAME 2 "--hz 5 --lights LW,LZ -- --quiet-light ..."`):

| per pass of the six rides | 5 Hz phase 2 | 5 Hz phase 0 | `--lights-signs` | `-keep` | `-keep -nohits` | `-nohits` (starts allowed) |
|---|---|---|---|---|---|---|
| false rules | 23 | 25 | 24 | 25 | **23** | 25 |
| missed intervals | 36 | 40 | 37 | 36 | 37 | 36 |
| matched intervals | 59 | 58 | 57 | 56 | 60 | 61 |
| momentary found / engine events | 109 / 211 | 108 / 207 | 112 / 233 | 113 / 231 | 111 / 222 | 111 / 219 |
| warnings found / engine events | 36 / 83 | 37 / 90 | 39 / 99 | 41 / 98 | **41 / 94** | 41 / 94 |
| interval start delay, median s | -0.5 | -0.4 | -1.5 | -1.0 | -0.1 | -0.2 |
| priority_road agreement | 0.923 | 0.933 | 0.894 | 0.913 | 0.923 | 0.923 |
| no_stopping agreement | 0.782 | 0.780 | 0.765 | 0.771 | 0.778 | 0.782 |
| one_way agreement | 0.805 | 0.805 | 0.836 | 0.790 | 0.813 | 0.814 |

What the light detector finds that the sign detector does not (krakow, published variant, tracks with no sign
detector box near any of their boxes, 32 tracks, contact sheet looked at by eye): mostly the backs of signs,
street-name and direction boards, shop signs, pharmacy crosses, advertising columns and barrier boards
(classified `other-sign` or no confident class, which the rules ignore), some real signs (a no-right-turn,
chevrons, plates), and a few backs of round signs read as round signs (a no-stopping on a back, 21 hits): the
false-positive risk the keep-only option removes.

Verdict: the reference says this is the largest free gain (signs listed 0.74 -> 0.81, first listing ~0.4 s
earlier, at no detector cost), the human labels say "within the noise, +3-5 warnings, more engine events".
**Not a default; A/B on the phone** with `--lights-signs-keep --lights-signs-nohits` first (equal false rules,
+5 warnings found), then `--lights-signs`. Cost: no detector run; per light frame 1.3-1.6 boxes (krakow,
score > 0.35) go through the sign classifier on the light thread (big cores): ~7-8 crops/s at 5 Hz. The phone
cost per crop is not measured here (L12 measured 25.4 ms per frame for 1.9 crops on the little cores, i.e.
up to ~13 ms per crop there; on the A75 a fraction of that): **measure `sg-lights` CPU on the phone**.

Limits: the C++ light thread decodes class 9 boxes clamped to the frame; the replay's cache clamps them to the
light view (boxes reaching below the wide view's lower edge differ). In signd the light frames do not
interleave perfectly with the sign frames, and late light updates (most of them) publish no message: the
"quiet" rows assume all are late.

## 4. Light detector

(see below; filled from the light runs)

## 5. Arrows as a light state

Data on disk: the car's recorder sessions (`~/roadsense-rec`, 14 sessions, 12,700 preview frames at 1 per
second, 2026-11-12 to 2026-11-18) and L21's 320 labelled detections of 2026-11-17 (`~/l21/sheets/labels.txt`:
r 100, g 90, ry 4, y 1, other 125). None is labelled as an arrow; L21 noted about five green arrows among the
320 by eye. BDD100K (the detector's and the state model's data) has no arrow attribute. So nothing can be
trained or evaluated now. What is needed: crops of lit arrow signals (left / straight / right, red / yellow /
green), from the car's own camera, at least ~100 per direction and colour for a first classifier and ~300 for
a usable one, plus the same number of full-disc lights from the same drives; at 1 recorded frame per second and
~5 arrows per 320 detections that is many hours of town driving with the recorder, or a recorder mode that
keeps full-rate crops of light tracks. A licence-clean public set with arrow labels was not found on disk
(DriveU DTLD has arrow pictograms but is research-only like BDD100K; not downloaded).

## 6. What was built (branch v2-l24s)

* `signd/yolo.{h,cc}`: `PipelineConfig::roi2`, `roi2_every`; `SignPipeline::process(..., view)`, `view_for`,
  `view_rect`; `SignPipeline::classify()` (thread-safe classifier for boxes from elsewhere).
* `signd/tracker.{h,cc}`: `update(..., seen)` (tracks outside the view are not aged; cap 3 x max_age);
  `Detection::external`, `TrackerConfig::external_starts`, `external_hits`.
* `signd/lights.{h,cc}`: `LightsConfig::sign_cls`, `sign_thr`; `LightPipeline::sign_boxes()`.
* `signd/main.cc`: `--roi2`, `--roi2-every`, `--lights-signs`, `--lights-signs-thr`, `--lights-signs-keep`,
  `--lights-signs-nohits`; `view` in the message only with a second view (0/1) or light updates (2).
  The rate-limit loops are untouched.
* `tools/signd_replay.cc`: `--roi2`, `--roi2-every`; `tools/lights_replay.cc`: `--both` (both light views on
  every frame).
* `tools/signs_track_replay.cc`, `tools/signs_tracker_test.cc` (ctest `signs_tracker`), `tools/signs_eval/`.

Default path: `signd_replay` of this branch and of `v2` (b50c08a) give identical output apart from `detect_ms`
on 51 krakow frames (9 ended tracks) and 60 frames of `ljkr.nv12`. ctest: 2/2. signd itself was run on the
laptop against a frame ring (`tools/ring_replay.py` of roadsense, `ljkr.nv12`) with `--roi2` and
`--lights-signs`: views 0/1/2 alternate, seq monotonic; ljkr has no sign track, so that run checks plumbing only.

## 7. Phone A/B (pending: the phone was not reachable)

Add to the unit's `ExecStart` (today's arguments unchanged), one variant at a time, and compare against the
unit as is on the same replay ride:

* light proposals, safest: `--lights-signs-keep --lights-signs-nohits`
* light proposals, most recall: `--lights-signs`
* (not recommended) second sign view: `--roi2 0.296875,0.2666667,0.796875,0.8`

Measure: `sg-lights` and `sg-signs` CPU (L21 `measure.py`), the stats line `lights-signs: N light updates older
than the last message`, road runs/s, `signs` messages/s.
