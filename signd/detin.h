// Detection inputs for a tracker: what the light thread's road users carry besides box, class and score
// (signd --lights-ru-low / -flags / -view / -emb; docs/DETECTION_INPUTS.md). Every part is off by default and
// adds only optional keys to the `road_users` entries (or the optional list `road_users_low`).
//
//   flags   (ru_flags)    which image edges cut the box (`trunc`), how much of it other, nearer boxes cover
//                         (`occ`, by whom `occ_by`, which of its edges `occ_edges`): a consumer stops trusting
//                         that side of the box
//   view    (ru_view)     from the pixels of the box, no model: left-right mirror symmetry (a rear or a front
//                         view is symmetric, a side or three-quarter view is not), the lamp pair (two bright
//                         blobs at one height, red = rear, white = front), the ground contact segment
//   emb     (EmbPooler)   an appearance vector per box, pooled from a feature map of the detector's own
//                         network (a tap model), for re-identification after an occlusion
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "yolo.h"

namespace roadsense {
namespace signs {

struct RoadUserDet;

// ---- flags ---------------------------------------------------------------------------------------------
// Edge bits as RoadUserDet::clip: 1 left, 2 top, 4 right, 8 bottom.
//
// Which of two overlapping boxes is nearer: on a common ground the one whose bottom edge is lower in the image.
// A bottom cut by the view (clip b) or the frame is lower than it shows: such a box is nearer than one whose
// bottom is seen; two cut bottoms, or bottoms within `tie` of each other, are not ordered, and each box then
// counts the other as covering it (the flag errs towards "do not trust this box's size").
struct RuFlagsConfig {
    float edge_px = 2.f;      // a box edge within this of the frame edge is cut by it
    float tie_px = 2.f;       // bottoms nearer than max(tie_px, tie_rel x the taller box) are not ordered
    float tie_rel = 0.04f;
    float edge_hidden = 0.5f; // share of an edge's length inside nearer boxes from which the edge counts as hidden
};
// Fills trunc, occ, occ_by, occ_edges of `users`; occluders are the boxes of `users` (occ_by = index into it).
// `low` (optional): boxes of the second, low-score list: flagged against `users`, never occluders themselves.
void ru_flags(std::vector<RoadUserDet> &users, std::vector<RoadUserDet> *low, int frame_w, int frame_h,
              const RuFlagsConfig &cfg = RuFlagsConfig());
// +1: a is nearer than b, -1: b is nearer, 0: not ordered. Exposed for tests.
int ru_nearer(const RoadUserDet &a, const RoadUserDet &b, const RuFlagsConfig &cfg = RuFlagsConfig());

// ---- view cues from the pixels of the box -----------------------------------------------------------------
// Lamp pair: at night a vehicle is two bright blobs at one height on a dark body. Their centroids do not move
// with the blooming (a blob grows about its centre) or with the blur of the car's own shaking (both blobs
// shift alike), so their separation is a ruler on the object: its change is the scale rate (signd/scale.h), its
// size a range cue (lamp centres of a car are 1.1-1.5 m apart), its colour says which end faces the camera.
struct LampPair {
    bool ok = false;
    float xl = 0, yl = 0, xr = 0, yr = 0;  // centroids of the left and the right blob, frame px
    float sep = 0, sep_std = 0;            // their distance and its 1-sigma, frame px
    int colour = 0;                        // 0 not known (amber, mixed), 1 red (rear), 2 white (front)
};
struct ViewConfig {
    int max_w = 96, max_h = 64;   // working crop: the box is box-averaged by an integer factor to fit
    float night_luma = 90.f;      // lamps are searched only in a frame whose median luma is at most this (ru_view())
    float max_bg = 160.f;         // median luma of the box above this: no lamp search (glare lifts it to ~150)
    float min_peak = 150.f;       // 98th percentile of the box's luma
    float min_contrast = 60.f;    // peak - median
    float cen_floor_px = 0.25f;   // floor of a centroid's 1-sigma, frame px (blob shape changes)
    float sep_rel = 0.02f;        // ... and this share of the separation, in its 1-sigma
    int sym_min_w = 16, sym_min_h = 8;  // boxes smaller than this get no symmetry
};
const char *lamp_colour_name(int c);
// The lamp pair of a vehicle box; false (out.ok false) when the box is not a dark scene with two matching blobs.
bool ru_lamps(const Nv12View &f, const RoadUserDet &u, LampPair &out, const ViewConfig &cfg = ViewConfig());
// Left-right mirror symmetry of the box's luma: the normalised correlation of a 32 x 16 reduction of the box with
// its mirror image, the best over axis shifts of +- 1/8 of the width. Near 1: a rear or a front view; lower: a
// side or three-quarter view, or an object half hidden. < -1: not measured (box too small).
float ru_symmetry(const Nv12View &f, const RoadUserDet &u, const ViewConfig &cfg = ViewConfig());
// Median luma of the frame on a coarse grid (every 8th pixel of every 8th row).
float frame_median_luma(const Nv12View &f);
// Fills the view cues of vehicles (sym, lamps) in place.
void ru_view(const Nv12View &f, std::vector<RoadUserDet> &users, const ViewConfig &cfg = ViewConfig());

// ---- appearance vector ------------------------------------------------------------------------------------
// For re-identification after an occlusion: a short vector per box such that two boxes of one object are near
// and boxes of different objects far (Euclidean distance of the published int8 values).
//   "yuv3x3"  no model: the central 90 % of the box in 3 x 3 cells, the mean Y, Cb, Cr of each (27 numbers:
//             nine Y, nine Cb, nine Cr; Y in units of 50 grey levels about 110, chroma in units of 20 about 128,
//             published x 32 as int8). On the ride library it re-identifies better than any pooled feature map
//             of the detector (docs/DETECTION_INPUTS.md).
//   "reid32+yuv3x3"  a small network on the 32 x 32 crop (tools/reid/train_reid.py): 32 numbers, L2-normalised,
//             published x 127 as int8, FOLLOWED by the 27 yuv3x3 numbers (x 32): 59 values. The distance to use is
//             sqrt(|d learnt / 127|^2 + (0.5 |d yuv / 32|)^2): on held-out rides the two together re-identify
//             better than either (LightPipeline runs the network; this header only holds the crop geometry).
constexpr int kEmbMax = 64;
struct Appearance {
    int n = 0;                 // 0: none
    int8_t v[kEmbMax] = {};
};
// The yuv3x3 vector of a box; false (out.n = 0) for a box under 8 px on a side.
bool ru_appearance_yuv(const Nv12View &f, const RoadUserDet &u, Appearance &out);
// The box with a 5 % margin as planar RGB / 255, 32 x 32 (the learnt embedding's input); false when too small.
bool ru_appearance_crop(const Nv12View &f, const RoadUserDet &u, float *rgb_3x32x32);
std::string appearance_hex(const Appearance &a);

}  // namespace signs
}  // namespace roadsense
