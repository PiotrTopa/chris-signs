# Detection inputs for a tracker (`signd --lights-ru-low / -flags / -view / -emb / -scale-lamps / -scale-multi`)

Date: 2026-11-24 (roadsense lane `detin`). The light thread's road users (`--lights-road-users`,
[LIGHT_MODELS.md](LIGHT_MODELS.md)) were a box, a class and a score, since 2026-11-23 with an image scale rate.
A tracker downstream (chris-integrator's radar) gets its remaining wrong answers from what such a box leaves
out: which boxes are whole, which objects were nearly detected, what an object looks like, which end of a
vehicle faces the camera, and how fast it grows in the dark. This file records what was added, why in this
form, and what was measured. The message contract is in roadsense `docs/ARCHITECTURE.md` (light provider); how
a tracker should take each field: roadsense `docs/lanes/detin-consumption-design.md`.

Everything is **off by default** and adds only optional keys. With all options off the messages are byte for
byte what they were (checked with the final binary: every message line of two rides, 3000 passes, a day and a
night ride, against the outputs of commit 12a960e, the time fields `lights_ms` and `timing` set aside; and with
everything on the first list is the same, box for box, on five rides). Code: `signd/detin.h`, `signd/detin.cc`, `signd/scale.cc`; no new dependency, no new model in the
default configuration.

| option | keys | what it costs (laptop, one thread; section "Cost") |
|---|---|---|
| `--lights-ru-low X` | `road_users_low` (list) | part of the decode |
| `--lights-ru-flags` | `trunc`, `occ`, `occ_by`, `occ_edges` | 0.3-0.7 microseconds per box |
| `--lights-ru-view` | `sym`, `lamps` | 9-56 (symmetry) + 9-140 (lamps, dark frames only) microseconds per vehicle box |
| `--lights-ru-emb yuv` | `emb`, message key `emb_kind` | 1.4-8.4 microseconds per box |
| `--lights-ru-emb FILE.onnx` | the same, with a small learnt part | 0.63 ms per box (one ORT run), at most 16 boxes a pass |
| `--lights-ru-scale-lamps` | `scale_src` (and more boxes with `scale_rate`) | 0.03-0.17 ms per box with a lamp pair (dark frames only) |
| `--lights-ru-scale-multi N` | (more boxes with `scale_rate`) | the patch alignment x 2.5 at N = 3 |

## 1. The low-score list (`--lights-ru-low X`)

The detector's output is thresholded per class (0.35-0.50). An object the detector half sees (far, dark, partly
hidden) is usually still in the output below the threshold. With `--lights-ru-low X` the boxes scoring above
`X` but not above their class threshold are listed too, in a list of their own:

* the first list is computed exactly as before (thresholds, class-agnostic NMS 0.5, rider pairing) and does not
  change with the option;
* a low box is dropped when a road user of the first list (after the rider pairing) or a better low box overlaps
  it by more than the NMS limit; at most 20 low boxes a pass (`ru_low_max`), best first; riders are not paired;
* the list is a key of its own, `road_users_low`, not a flag on entries of `road_users`: a consumer that does
  not know it sees no new boxes. Its entries have the same keys (also `scale_rate`, flags and `emb` when those
  are on; `occ_by` indexes `road_users`). While the governor sheds the road users both lists are absent.

A tracker offers these boxes to objects it already holds and never starts one from them (ByteTrack's second
association stage; here the detector's half of it).

### Measured against labels (BDD100K val, `tools/road_users/detin_bdd.py`)

3000 images (every third of the first 9000 file names), the whole frame through the 384x640 export of the same
network (the unit's 320x960 views cut most road users of a BDD picture at the bottom), first list at the unit's
thresholds, 37 488 labelled road users (25 456 labelled occluded, 15 236 under 30 px tall). A box is "on" a
label at IoU >= 0.5.

| | low boxes per image | on a label the first list missed | on a label it found | on none | labels found | ... labelled occluded | ... under 30 px |
|---|---|---|---|---|---|---|---|
| first list alone (18 143 boxes, 96.4 % on a label) | | | | | 46.6 % | 34.3 % | 15.3 % |
| + low, floor 0.30 | 1.46 | 68.3 % | 1.6 % | 30.1 % | 54.6 % | 43.5 % | 25.4 % |
| + low, floor 0.20 | 3.44 | 56.2 % | 2.8 % | 41.0 % | 62.1 % | 52.3 % | 36.1 % |
| + low, floor 0.15 | 4.80 | 49.1 % | 3.1 % | 47.8 % | 65.5 % | 56.4 % | 41.0 % |
| + low, floor 0.10 | 6.71 | 41.1 % | 3.4 % | 55.4 % | 68.7 % | 60.1 % | 45.6 % |
| + low, floor 0.05 | 10.04 | 31.6 % | 3.8 % | 64.6 % | 72.0 % | 64.0 % | 49.7 % |

So at a floor of 0.10 the low list holds 41 % of the road users the first list misses (of the small ones the
share found triples), and a little over half of its boxes are on no label ("on none" counts boxes that sit on a
real object with IoU under 0.5 too). How good a low box is, where it is on a label (robust sigma of the edge
error as a share of the label's height): 10-13 % against 3.8-4.0 % for the first list (in pixels at like sizes:
2.3-4.7 against 1.9-3.7), median IoU 0.67 against 0.87, class right 95 % against 98 %. The floor 0.05 hits the
20-box cap in busy pictures.

### Measured on rides, the unit's views (`tools/road_users/low_check.py`, no labels)

Objects = chains of first-list boxes; a gap = a pass between two boxes of one chain where the first list has no
box of it although the pass's view covers the place. Floor 0.10:

| ride (5 minutes at the unit's schedule) | first-list boxes a pass | low boxes a pass | gap passes with a low box on the object | ... at floor 0.20 | low boxes on no first-list object |
|---|---|---|---|---|---|
| warsaw-city (day) | 0.48 | 1.22 | 23 of 34 (68 %) | 20 of 34 (0.40 low boxes a pass) | 93 % |
| a1-night-fog | 0.30 | 1.04 | 70 of 93 (75 %) | 51 of 93 (0.33) | 87 % |
| krakow-night-tunnel | 0.16 | 0.55 | 6 of 7 | 6 of 7 (0.17) | 95 % |
| gdansk-night-snow | 0.05 | 0.37 | 4 of 4 | 3 of 4 (0.12) | 95 % |
| a1-night | 0.03 | 0.62 | no gap pass | (0.19) | 99 % |

Where the first list loses an object it has before and after, a low box is on the object in two passes of
three or more: that is what a second association stage picks up. Most low boxes, though, are on nothing the
first list ever shows; without labels on these rides it is not known how many of them are real objects the
detector never boxes above the threshold (BDD100K says 41 % of the low boxes are on a labelled road user the
first list missed) and how many are clutter. A tracker must take the list for what the contract says: a way to
continue what it holds, never a reason to start.

## 2. Flags (`--lights-ru-flags`)

* `trunc`: the FRAME edges within 2 px of the box (`l t r b`). `clip` (since lane L42) names the edges of the
  view's region; a box cut by the bottom of the wide view (`clip` b) may be whole in a lower view, a box at the
  frame's edge is cut for good. Present on every box when the flags are on (possibly empty).
* `occ`, `occ_by`, `occ_edges`: the share of the box covered by NEARER boxes of the first list (union, on a
  16 x 16 grid of the box), the index of the one covering most, and the edges of this box of which at least half
  lies under nearer boxes. Only when something covers the box.
* Nearer: on a common ground the box whose bottom edge is lower in the picture. A bottom cut by the view or
  the frame is lower than it shows, so such a box is nearer than one whose bottom is seen. Two cut bottoms, or
  bottoms within max(2 px, 4 % of the taller box), are not ordered: each box then counts the other. The flag
  therefore errs towards "covered": it is there to take trust away from a box's size, not to state a depth
  order.
* Low boxes are flagged against the first list and are never occluders.

What it cannot know: anything that is not a box (a pole, a tree, a kerb, an unboxed vehicle).

### Measured (BDD100K val, the labels' own `occluded` / `truncated` attributes)

| | flagged | of the flagged, labelled so | of the labelled, flagged |
|---|---|---|---|
| `trunc`, computed on the label boxes (37 488) | 2814 | 98.3 % | 86.5 % |
| `occ` > 0, on the label boxes | 20 116 | 93.7 % | 74.1 % |
| `trunc`, the detector's boxes on a label (17 481) | 2384 | 97.9 % | 94.5 % |
| `occ` > 0, the detector's boxes | 6801 | 93.0 % | 72.5 % |

By amount, on the label boxes: `occ` 0: 38 % labelled occluded; (0, 0.1): 80 %; [0.1, 0.3): 92 %; >= 0.3: 97 %.
Boxes with neither flag: 40 % are labelled occluded all the same (BDD labels a road user occluded for any
cover) and 2 % truncated. **A flag means "do not trust this side"; no flag does not mean "whole".** Which of
two overlapping boxes is nearer has no label to be held against; unit checks: `build/detin_test`.

## 3. View cues (`--lights-ru-view`): lamp pair and mirror symmetry

From the box's own pixels in the full-resolution frame, vehicles only (car, truck, bus, train), no model.

### `lamps`
In a dark frame (median luma of the frame, every 8th pixel, at most 90) a vehicle is two bright blobs at one
height. Per box: luma box-averaged to at most 96 x 64; background = median, peak = 98th percentile; searched
when the peak is at least 150 and 60 above the background; blobs = connected pixels above the half level;
centroid weighted by the luma above that level; colour from the chroma of the blob's halo (the core is clipped:
`red` for Cr >= 14 above neutral and not bluer than red, `white` for next to none, else `unknown`). The pair:
two blobs whose heights differ by under 12 % of their distance, 0.3-0.98 of the box width apart, about the box's
middle (unless a side is cut), areas within a factor of 10, never one red and one white; of several, the pair
with the most light in its weaker blob. Reported: colour, the two centroids, their distance `sep` and its
1-sigma `sep_std` = hypot(0.25 px per centroid, 2 % of the distance).

Why centroids: a lamp's blob grows with the exposure and the glare (blooming) about its own centre, and the
blur of the car's shaking moves both blobs alike. The box of the same vehicle follows the glare.

Measured: a pair on 142 of 195 vehicle boxes of the krakow-night-tunnel ride at the unit's schedule (red 93,
white 30, unknown 19), 22 of 33 on a1-night, 2 of 629 on warsaw-city by day (the frame gate; without it bright
paint and sky reflections would pair up). By eye on a sheet of 48 night boxes (every fourth of the ride's): 30
pairs, 28 on the vehicle's two lamps with the right colour (red on tail lamps, white on headlamps), one on a
lamp and a reflection on the body, one on two lights of the background (colour `unknown`); the boxes without a
pair were dark side views, glare that filled the box and cars in the lit tunnel. The noise figures come from the same ride (scatter of ln `sep` about its local
trend over three passes, 54 triples): 0.31 px at separations of 12-20 px, about 2 % above 40 px, 0.8 px under
12 px, where the pair picked flips between the lamps and their reflections: **use `sep` only from 10 px up**
(the scale rate does). No ground truth for the colour beyond the sheet.

### `sym`
The normalised correlation of the box's luma (reduced to 32 x 16) with its mirror image, the best over nine
positions of the axis (+- 1/8 of the width). For boxes of at least 16 x 8 px.

**Not validated.** The intent was a cue for "seen from behind or in front" against "seen at an angle". On the
ride views the only clean day boxes (16, cars crossing on a bridge, all side views) scored 0.02-0.89: a car's
side is itself close to mirror-symmetric at this resolution, so `sym` alone does not tell a rear from a side;
the box's aspect does that, and `sym` can at most separate three-quarter views. No labelled set with the
vehicle's orientation is on this machine (section 6).

## 4. Appearance vector (`--lights-ru-emb`)

For re-identification after an occlusion. Three sources were compared before anything was built into signd.

### Where the vector should come from: evidence

Identities without labels: on rides at 20 fps a box continues a chain when it is the mutual best IoU >= 0.5 of
the chain's last box and neither has a second candidate above 0.3; a chain is one object. Score: a query box
and its own box 1, 2, 4 s later (the unit's 5 Hz), against every box of every OTHER chain of the ride; rank-1
share and AUC.

**(a) A tap on the detector's feature maps** (`<datasets>/detin-lane/exp/reid_extract.py`, not committed: it
needs the research-licensed `.pt`): RoI-pooled (inner 80 % of the box, 2 x 2 samples) outputs of the backbone
(layers 1, 2, 4, 6), the neck (16, 19) and the detect head's box / class branches, raw, standardised per frame,
standardised with another ride's statistics, 2 x 2 grids, PCA to 32 / 16 / 8 (from the other ride), int8.
Wide view, 5 minutes each of warsaw-city (day, 4451 boxes, 69 chains of ten or more boxes) and
krakow-night-tunnel (1550 boxes, 27 chains):

| vector | day: rank-1 at 1 / 2 / 4 s | day: AUC | night: rank-1 | night: AUC |
|---|---|---|---|---|
| best tap: backbone layer 6 (128 numbers, stride 16, raw) | 0.66 / 0.72 / 0.80 | 0.941 / 0.930 / 0.939 | 0.43 / 0.24 / 0.03 | 0.965 / 0.932 / 0.842 |
| backbone layer 4 (128, stride 8, raw) | 0.61 / 0.69 / 0.76 | 0.928 / 0.922 / 0.930 | 0.38 / 0.13 / 0.06 | 0.957 / 0.914 / 0.809 |
| layer 2 (64, stride 4, raw) | 0.59 / 0.69 / 0.76 | 0.904 / 0.920 / 0.940 | 0.37 / 0.18 / 0.11 | 0.960 / 0.927 / 0.901 |
| neck P3 (layer 16, 64) | 0.56 / 0.64 / 0.71 | 0.916 / 0.901 / 0.905 | 0.24 / 0.05 / 0.00 | 0.928 / 0.858 / 0.750 |
| layer 4 -> PCA 32, int8 | 0.59 / 0.68 / 0.74 | 0.922 / 0.916 / 0.933 | 0.22 / 0.11 / 0.05 | 0.920 / 0.870 / 0.802 |
| **3 x 3 cells of mean Y, Cb, Cr of the crop (27)** | **0.70 / 0.74 / 0.82** | **0.945 / 0.961 / 0.979** | **0.48 / 0.23 / 0.09** | 0.959 / 0.920 / 0.891 |
| luma thumbnail 12 x 12 | 0.65 / 0.73 / 0.80 | 0.933 / 0.953 / 0.973 | 0.40 / 0.28 / 0.20 | 0.957 / 0.911 / 0.844 |

(The day gallery is 1044 boxes, the night one 388, so single differences of a few points are within the
noise; the neck's P4 map and the detect head's box / class branches score below layer 6; standardising a tap
per frame or with another ride's statistics changes little; concatenating a tap with the colour cells did not
beat the cells.) **No tap on the detector's feature maps beats nine coloured cells of the picture**: the best
one comes within a few points of rank-1 and stays behind in AUC by day (0.93-0.94 against 0.95-0.98); the
detector's features are trained to give every car the same answer. A tap would also need a second export of
the network with an extra output and the same change in the ncnn conversion. Not built.

**(b) The model-free vector, `yuv3x3`** (built: `--lights-ru-emb yuv`): the central 90 % of the box in 3 x 3
cells, per cell the mean Y, Cb, Cr from the full-resolution NV12 frame (at most about 16 x 16 samples a cell),
27 numbers: nine Y as (Y - 110) / 50, nine Cb and nine Cr as (C - 128) / 20, published x 32 as int8 (hex).
On the full-frame crops of two rides held out from (c) (`tools/reid/`, 384x640 export, every 2nd frame):

| ride (held out) | score | `yuv3x3` at 1 / 2 / 4 s | learnt 32 | learnt 32 + `yuv3x3` (as published) |
|---|---|---|---|---|
| warsaw-city (day, 10 min, 21 022 vehicle boxes) | rank-1 against the other boxes of the frame (4.1 on average) | 0.948 / 0.922 / 0.883 | 0.968 / 0.945 / 0.920 | **0.975 / 0.954 / 0.931** |
| | ... AUC | 0.980 / 0.970 / 0.955 | 0.989 / 0.980 / 0.970 | 0.991 / 0.982 / 0.974 |
| | rank-1 against every box of every other chain of the ride | 0.605 / 0.469 / 0.353 | 0.616 / 0.491 / 0.395 | **0.719 / 0.598 / 0.463** |
| | ... AUC | 0.984 / 0.974 / 0.958 | 0.992 / 0.985 / 0.974 | 0.993 / 0.987 / 0.976 |
| krakow-night-tunnel (night, 10 min, 1844 vehicle boxes) | rank-1 in the frame (1.0 other box: chance 0.5; 150 / 130 / 116 queries) | 0.900 / 0.808 / 0.750 | 0.940 / 0.862 / 0.862 | 0.907 / 0.869 / 0.802 |
| | rank-1 against the ride | 0.496 / 0.346 / 0.169 | 0.484 / 0.388 / 0.221 | **0.621 / 0.455 / 0.279** |
| | ... AUC | 0.970 / 0.953 / 0.924 | 0.969 / 0.938 / 0.915 | 0.978 / 0.964 / 0.946 |

"In the frame" is the association a tracker faces (the boxes of one frame are different objects, so the score
is clean); "the ride" is harder and counts another chain of the same car as a miss, which holds every vector
down alike. By box width at 2 s against the ride, `yuv3x3`: rank-1 0.19 under 24 px, 0.32 at 24-48, 0.47 at
48-96, 0.61 above (day): the vector is weakest where the tracker's range is worst.

Variants on the same crops: 2 x 2 cells are clearly worse, 4 x 4 (48 numbers) gain 3 points of rank-1 at the
same AUC, removing the mean luma or the luma altogether costs, the int8 form loses nothing.

**(c) A small learnt head** (built: `--lights-ru-emb FILE.onnx`; `tools/reid/train_reid.py`): 32 x 32
RGB crop -> four conv blocks (16, 32, 64, 96 channels) -> global mean -> 32 numbers, L2-normalised (214 k
parameters), published x 127 as int8. Trained without labels: identities are the chains of the training
rides, positives two crops of one chain, supervised contrastive loss over batches of 48 chains x 4 crops,
augmentation by shift, scale, brightness, contrast, gamma, channel gain, blur and noise.
Training set: 46 470 crops of 1470 chains (at least four boxes over half a second, not covered by another
box by 0.3 or more) from 8 rides of the ride library, every 2nd frame of their first 10 minutes
(forest-lubelskie, gdansk-night-snow, katowice-rain, krakow-2024, krakow-lights, rural-zloczew, s1-roadworks,
s7-elblag-lowsun); 2500 iterations on the CPU, about 15 minutes; 214 032 parameters. The two test rides
(warsaw-city, krakow-night-tunnel) took no part. The weights come from YouTube rides under CC BY and are kept
local like the other trained files (`<datasets>/detin-lane/out/reid32.onnx`); nothing was downloaded for it.

Result (table above): the learnt vector alone is a little better than `yuv3x3` in the frame and about equal
against the ride; **the two together are better than either** (day, in the frame: the misses fall from 5.2 /
7.8 / 11.7 % to 2.5 / 4.6 / 6.9 %; against the ride rank-1 0.72 against 0.61). So with a model file signd
publishes both: the 32 learnt numbers followed by the 27 (`emb_kind` "reid32+yuv3x3"; distance
sqrt(|d learnt / 127|^2 + (0.5 |d yuv / 32|)^2)). signd's own crop path gives the same vector as the training
path to a cosine of 0.995 (median over 109 boxes of a ride; 0.93 at worst).

The answer to "tap or dedicated head": neither first. The model-free cells cost microseconds and beat every
tap; the small head roughly halves the day-time misses on top and costs one network run per box (section 7).
Distance tables of both kinds for a tracker's likelihood ratio (quantiles and log-normal fits of same-object and
other-object distances at 1, 2, 4 s, day and night): roadsense `docs/lanes/detin-emb-distances.json`
(`reid_eval.py --dist`). With those fits and a ratio clamped to [1/8, 8], by day 79 % (`yuv3x3`) / 86 %
(with the head) of the other-object pairs get a ratio of 1/4 or less at 1 s (68 / 77 % at 4 s) while 3-5 % of
the same-object pairs do; at night 23-34 % against 5-6 %: in the dark the vector says little.

### What signd does
`--lights-ru-emb yuv`: `emb` on every box of both lists of at least 8 px on a side; `emb_kind` "yuv3x3".
`--lights-ru-emb FILE.onnx`: one ONNX Runtime run per box (one thread), the largest boxes of the first list
first, then the low list, at most 16 a pass (`--lights-ru-emb-max`), the 27 `yuv3x3` values appended;
`emb_kind` "reid32+yuv3x3"; boxes beyond the cap carry no vector; a missing file falls back to `yuv` with a line
on stderr (as the state CNN does). Vectors of different kinds are not comparable.

## 5. Scale rate in the dark (`--lights-ru-scale-lamps`, `--lights-ru-scale-multi N`, and the low list)

### Why the scale rate is worst at night
On the night ride (krakow-night-tunnel, 195 vehicle boxes at the unit's schedule) the patch alignment of
2026-11-23 measured 70 boxes (36 %; by day 45 %) with a stated std of 0.046-0.056 /s (by day 0.009):

* 58 boxes were the first of their chain and 8 had no earlier patch in the window: the detector misses in runs
  at night, a chain breaks after 1.2 s without a box, and every restart costs 0.2-1.0 s without a value;
* 35 were measured but refused for a stated std above the limit, 11 for a poor match, 6 for a poor fit: a
  vehicle in the dark is two lamps on a dark body (sheets in the lane's scratch): next to no texture besides
  the lamps, and the lamps are clipped blobs whose size follows the exposure, not only the distance;
* the pass rate is 5 a second at best and about half the passes box a given object.

### What was added
* **The lamp pair as a second measurement** (`--lights-ru-scale-lamps`). Per chain the estimator keeps the
  lamp separation of every pass that had a pair (section 3). The rate is the slope of a weighted line through
  ln `sep` over the passes of the window (at least three, spanning at least 0.2 s, separations of at least
  10 px, one colour, no step above 0.25), Huber weights re-estimated three times (one pass on other blobs must
  not turn the line); the stated std is the fit's, grown by its own scatter (chi^2 per degree of freedom, never
  shrunk), and for a value from the lamps alone times 1.5 (measured below). `scale_dt_s` is the fit's span.
* **Combination.** A box with both values gets their inverse-variance mean, `scale_src` "both"; the two share
  pixels, so the combined std is not taken below 0.7 of the better one's; where they differ by more than three
  of their joint sigma, half the difference is added to the std. A box with only the lamp value: `scale_src`
  "lamps". Patch only: no `scale_src` (as before).
* **Several baselines** (`--lights-ru-scale-multi N`): the current frame is aligned with up to N earlier
  patches of the window instead of the oldest only; the rate is the generalised least-squares slope of the log
  scale changes over their baselines with the errors correlated by 0.5 (all share the current frame); one
  baseline that disagrees with the others by more than 3.5 sigma is left out; the std grows with the scatter.
* **The low list continues chains** (with `--lights-ru-low`): a low box may continue (never start) a chain and
  is measured itself, so a miss run of the first list no longer restarts the chain.

### Synthetic (`build/scale_rate_test`, ctest `scale_rate`)
Night scene: a dark body (luma about 25, faint texture), two lamps with clipped cores whose halo width changes
from frame to frame (0.6-1.8 px), exposure +- 20 % per frame, 0-3 px of motion blur, red chroma, noise 4 grey
levels; 60 trials a case, values from 0.4 s on. This scene is harsher on the patch alignment than the recorded
night ride (there the patch values agree with each other within their stated std) and kinder to the lamp pair
(clean Gaussians), so it shows mechanisms, not the car's numbers:

| case | mode | coverage | bias (1/s) | scatter (1/s) | stated std (1/s) | rms pull | > 3 sigma |
|---|---|---|---|---|---|---|---|
| day far 16 px, closing 0.15/s | patch | 93 % | -0.0017 | 0.0125 | 0.0105 | 1.00 | 1.1 % |
| | patch x3 | 97 % | -0.0027 | 0.0161 | 0.0141 | 0.97 | 1.0 % |
| day far 24 px, contrast 15 | patch | 86 % | +0.0072 | 0.0287 | 0.0180 | 1.41 | 4.0 % |
| | patch x3 | 96 % | +0.0062 | 0.0301 | 0.0257 | 1.16 | 4.2 % |
| day mid 60 px | patch | 100 % | -0.0003 | 0.0027 | 0.0031 | 0.82 | 0 % |
| | patch x3 | 100 % | -0.0004 | 0.0025 | 0.0033 | 0.77 | 0 % |
| night far 16 px (lamps 13 px apart), closing 0.15/s | patch | 52 % | +0.0039 | 0.0499 | 0.0352 | 1.43 | 6.7 % |
| | lamps only | 83 % | +0.0106 | 0.0390 | 0.0968 | 0.44 | 0 % |
| | patch + lamps | 90 % | +0.0083 | 0.0451 | 0.0562 | 1.13 | 3.3 % |
| night far 16 px, still | patch | 58 % | -0.0051 | 0.0549 | 0.0387 | 1.60 | 4.3 % |
| | lamps only | 67 % | +0.0069 | 0.0417 | 0.0866 | 0.46 | 0 % |
| | patch + lamps | 86 % | -0.0012 | 0.0496 | 0.0518 | 1.25 | 2.9 % |
| night mid 32 px, closing 0.25/s | patch | 100 % | +0.0001 | 0.0144 | 0.0160 | 1.35 | 3.5 % |
| | lamps only | 100 % | -0.0003 | 0.0130 | 0.0698 | 0.16 | 0 % |
| | patch + lamps | 100 % | +0.0003 | 0.0107 | 0.0131 | 1.34 | 3.5 % |
| night mid 32 px, opening 0.20/s | patch | 84 % | +0.0015 | 0.0235 | 0.0279 | 1.15 | 3.0 % |
| | lamps only | 100 % | -0.0052 | 0.0167 | 0.0739 | 0.27 | 0 % |
| | patch + lamps | 100 % | -0.0013 | 0.0159 | 0.0275 | 1.04 | 2.5 % |
| night near 90 px, closing 0.35/s | patch | 100 % | -0.0004 | 0.0039 | 0.0054 | 0.91 | 0.5 % |
| | lamps only | 100 % | +0.0001 | 0.0005 | 0.0607 | 0.01 | 0 % |
| | patch + lamps | 100 % | -0.0004 | 0.0037 | 0.0052 | 0.90 | 0.5 % |

What it shows. More baselines buy coverage of far objects (86 -> 96 % at low contrast) and a more honest std
there (pull 1.41 -> 1.16), not precision. In the dark the patch alignment loses half the far boxes and its
stated std is too small against blooming and exposure changes (pull 1.4-1.6, 4-7 % beyond 3 sigma); the lamp
pair covers 67-100 % with no value beyond 3 sigma, its stated std (set by what real blobs do, section 3) well
above the scatter of these clean blobs; together 86-100 % of the far boxes get a value. On clean large blobs
the lamp pair is ten times more precise than the patch (0.0005 against 0.0039 /s at 90 px) and the combination
does not use that, because the stated lamp std is the recorded one: real blobs at that size change shape.

### Recorded rides, the unit's schedule (`lights_replay`, `tools/road_users/scale_check.py`)

Five rides cut to the unit's schedule (5 frame periods a second, wide / zoom alternating, idle pattern, 5
minutes each); first-list boxes with a value, then the low boxes with a value; "far" = box size under 40 px;
agreement = successive values of one object on disjoint baselines, in units of their stated std (1 = honest):

| ride | first-list boxes | patch alone (2026-11-23) | + lamp pair | + 3 baselines | + low list 0.10 | all three | far: median stated std, alone -> all | far: agreement, alone -> all |
|---|---|---|---|---|---|---|---|---|
| krakow-night-tunnel | 195 | 70 (36 %) | 93 (48 %) | 91 (47 %) | 75, + 22 low | **109 (56 %)**, + 34 low | 0.047 -> 0.042 /s | 0.68 (48 pairs) -> 0.83 (79 pairs), none beyond 3 |
| a1-night-fog | 334 | 162 (49 %) | 174 (52 %) | 193 (58 %) | 158, + 81 low | **196 (59 %)**, + 100 low | 0.025 -> 0.029 /s | 0.95 (113 pairs) -> 0.78 (135 pairs), none beyond 3 |
| gdansk-night-snow | 55 | 21 | 21 | 21 | 19, + 8 low | 20, + 9 low | 0.034 (6 values) | 4 pairs |
| a1-night | 33 | 7 | 7 | 7 | 7, + 14 low | 7, + 14 low | 0.034 (7 values) | 6 pairs |
| warsaw-city (day) | 667 | 259 (39 %) | 259 (no dark frame) | 275 (41 %) | 259, + 137 low | 275 (41 %), + 148 low | one far value | (near objects: real dynamics) |

On the two night rides with enough vehicles the share of first-list boxes with a value goes from 36 / 49 % to
56 / 59 %, and the values keep agreeing with each other within their stated std. The lamp pair's noise model was
set on krakow-night-tunnel; a1-night-fog took no part in that and shows the smaller gain from the lamps (+ 12
boxes) and the larger one from several baselines (+ 31). The stated std of a far box does not shrink (0.03-0.05
/s at night against 0.009 by day): the gain is coverage, not precision. The low list by itself does not raise
the first list's share (- 4 to + 5 boxes: a chain continued through a sloppy low box predicts the next box a
little worse); what it adds is a value on the low boxes themselves.

Patch and lamp value of the same box: on krakow-night-tunnel (17 boxes with both) the median difference is
-0.0003 /s with a robust sigma of 0.029 /s against a stated joint 0.063: no bias, one of the 17 lamp values off
by 3.4 joint sigma. On a1-night-fog (27 boxes) the lamp value is higher by 0.023 /s in the median (0.45 of the
joint sigma): glare in fog moves the blobs.

Lamp-only values (12 and 8 first-list boxes in the two rides; 37 and 44 more have both measurements): successive
lamp-only values of far objects disagreed by 1.58 (25 pairs) and 1.35 (20 pairs) of the line fit's std, with 12
and 5 % beyond 3 sigma: the fit does not see a pair of blobs that changes or glare that moves them. They are
therefore published with the fitted std times 1.5 (`lamp_only_std_x`): 1.05 and 0.90 on the same pairs, none
beyond 3 sigma. The factor is set on the two rides it is checked on.

### What it does not fix
The precision of a far box at night (0.03-0.05 /s) is unchanged: what was gained is coverage. The lamp-only
values rest on an empirical factor from two rides; a consumer should take `scale_src` "lamps" with a robust
likelihood. The remaining limit is the pass rate: three to five samples a second per object. The camera
delivers 20-30 frames a second and the ring holds six: measuring the lamp pair on the frames BETWEEN two passes
(the box interpolated, a blob search of some tens of microseconds) would give a line through 8-16 points a
second instead of 3-5 and make the robust fit robust. Designed, not built (the light thread reads one frame per
pass today).

## 6. Orientation and ground contact: what the present model gives, and what would be needed

The detector's output is a box and ten class scores. From it and the pixels, without training:
* which end faces the camera, in the dark (`lamps.c`);
* the ground contact row = the box's bottom edge, usable when the bottom is neither cut (`clip`, `trunc`) nor
  hidden (`occ_edges`): the flags say when NOT to use it. Turning the row into a range is the consumer's
  geometry (calibrated camera, pitch, the road's profile): roadsense `docs/lanes/detin-consumption-design.md`
  section 5. signd knows neither the calibration nor the road.

Not given by anything here: the heading (or even a rear / three-quarter / side class) by day, and the points
where the wheels meet the road. Rules on pixels were considered and not built: mirror symmetry does not
separate the views (section 3); the dark band under a vehicle against the road ("shadow edge") gives a contact
row to a fraction of a pixel by day, but it moves with the sun (a low sun throws the shadow towards the camera)
and there is nothing on this machine to hold its bias against.

**What a trained head would be** (the same pattern as the light-state CNN: a small network on the box's crop,
one run per box):
* input: the box + 10 % margin at 64 x 64; outputs: viewpoint as sin / cos of the observation angle (or 8
  bins + residual), the visibility and image position (relative to the box) of the two wheel contact points of
  the visible side and of the near face's bottom corners, and a truncation / occlusion score per edge;
* data: 3D box labels projected with the camera: **KITTI 3D object** (7481 training images with observation
  angle `alpha`, 3D dimensions and position, occlusion and truncation levels; CC BY-NC-SA 3.0, about 12 GB),
  **nuScenes** (1.4 M boxes with full orientation, CC BY-NC-SA 4.0, front camera subset about 50 GB) or
  **Cityscapes 3D**; for the wheel points **ApolloCar3D** (66 keypoints per car) or contact points derived from
  the projected 3D box bottom. None of them is on this machine and none was downloaded;
* how: crops from the label boxes with the detector's own box jitter (as `train_state2.py` does for lamps),
  then a few hundred labelled crops of the car's own camera for the domain gap, exactly as the light-state
  model was brought to this camera (LIGHT_MODELS.md);
* licence: the same status as the BDD-trained weights: research / non-commercial, never committed;
* cost: one run of a network of the state CNN's size per box, about 1 ms on a big core of the phone.

## 7. Cost

Laptop (i7 class), one thread, the time inside the light thread after the detector (nothing here touches the
GPU turn). **The laptop was loaded by other jobs while these were taken** (load average 8-30 on 12 cores): the
patch alignment of 2026-11-23 measured 0.24-0.81 ms per box in the same runs against 0.12-0.37 ms on the quiet
machine then, so the scale figures below are about twice a quiet laptop's; the microsecond figures moved
little between a load of 9 and of 25. Per box: `build/detin_test --bench` (median of 2001) and
`build/scale_rate_test` (the `ms/box` column); per pass: the `timing` of `lights_replay` on the rides.

| part | laptop, as measured | phone, estimated (big core 3-5 x a quiet laptop, little core about 10 x: the factors of LIGHT_MODELS.md) |
|---|---|---|
| low list (`--lights-ru-low`) | inside the decode; decode + flags + view + emb together per pass with road users: 0.09-0.12 ms median, 0.14-0.21 ms p90 against 0.06-0.09 / 0.09-0.10 with the options off | well under 1 ms a pass; the message grows by the low boxes (0.4-1.2 low boxes a pass on the rides) |
| flags | 0.3-0.7 us per box (0.006 ms for 12 boxes) | microseconds |
| `sym` | 9-13 us (21 x 16 px), 19 us (60 x 48), 32-56 us (180 x 120) per vehicle box | big 0.03-0.2 ms, little 0.1-0.5 ms per box |
| `lamps` | 9 us (21 x 16), 49-59 us (60 x 48), 140 us (180 x 120) per vehicle box, dark frames only; the darkness test 18 us a pass | big 0.03-0.7 ms, little 0.1-1.4 ms per box: 12 mid-sized vehicles in the dark about 2-3 ms (big) / 6 ms (little) a pass |
| `emb` yuv3x3 | 1.4-8.4 us per box | little core: under 0.1 ms per box |
| `emb` with the learnt part | 0.63 ms per box median, 0.92 p90 (ORT, one thread), cap 16 a pass: up to 10-15 ms a pass | about 3 ms per box on a big core (the light-state CNN: 65 k parameters, about 1 ms per crop on an A75; this network does about three times the arithmetic): up to 50 ms a pass at the cap, a quarter of a big core at 5 passes a second. Not measured on the phone |
| scale: lamp pair | 0.03 ms (16 px) - 0.17 ms (90 px) per box, lamp search included; with the patch: + 0.02 ms | big 0.05-0.4 ms per box |
| scale: 3 baselines | 2.5 x the patch alignment: 0.70 ms (16 px) - 1.5 ms (60 px) per box against 0.28-0.63 | big cores: 1-4.7 ms per box against 0.4-1.9 (LIGHT_MODELS.md), worst pass at the 12-box cap 32-55 ms against 13-22 |

Message size (all options on, floor 0.10): on the rides the mean `lights_replay` line (with its `raw` and `timing`
keys) grows from 331-530 to 452-675 bytes at night and from 1563 to 1886 by day (+ 20-45 %). Per box: `emb` 62
bytes (`yuv3x3`) or 126 (with the learnt part), `trunc` 11, the three `occ` keys about 45, `lamps` about 90,
`sym` 11; a low box is a whole entry of about 150 bytes.

Per pass on the rides (busy passes, everything on, floor 0.10, 3 baselines, lamps): scale part median 0.09-0.58
ms, p90 1.1-2.5 ms against 1.1-1.6 with the patch alone, the worst pass 3-5 ms (few vehicles a pass in these
views; the 12-box cap bounds a pass at about 2.5 times the 4.5 ms of LIGHT_MODELS.md).

No GPU work, no new thread, no new dependency. Memory: the scale chains gain three floats per pass and object.

## Reproduce

```sh
cmake --build build && ctest --test-dir build                       # detin, scale_rate, road_users, ...
# flags and low list against labels (BDD100K val as a FiftyOne export; 384x640 export of the detector)
lights_replay --backend ort --model bdd100k_yolo11n_384x640.onnx --in 640x384 --roi 0,0,1,1 --no-track \
  --road-users --ru-low 0.05 --ru-flags $(cat LIST) > bdd.jsonl
tools/road_users/detin_bdd.py samples.json bdd.jsonl LIST build/detin_test
# rides at the unit's schedule (frames <index>.jpg, every 4th of a 20 fps ride)
lights_replay --backend ort --model bdd100k_yolo11n_320x960.onnx --roi2 0.125,0.125,0.875,0.569 --thr-keep 0.12 \
  --coast 3 --idle ZW.Z.W --road-users --name-fps 20 --ru-scale --ru-scale-lamps --ru-scale-multi 3 --ru-low 0.10 \
  --ru-flags --ru-view --ru-emb yuv FRAMES > out.jsonl
tools/road_users/low_check.py out.jsonl
tools/road_users/scale_check.py out.jsonl [--road RIDE.road.jsonl.gz]
# appearance
tools/reid/crops_extract.py bdd100k_yolo11n_384x640.onnx RIDE.mjpeg ride.npz 0 11999 2
tools/reid/train_reid.py reid32 train-ride-*.npz
tools/reid/reid_eval.py --onnx reid32.onnx --size held-out-ride-*.npz
```
