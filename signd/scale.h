// Image scale change ("looming") of the road users of the light thread (signd --lights-ru-scale).
//
// For every reported road-user box the rate of change of the log size of the object's IMAGE is measured by
// aligning image patches, not by differencing box heights (Stein/Mano/Shashua 2003, Dagan et al. 2004): the
// box edges jitter by a few pixels, the texture inside the box does not. Range rate follows as
// Zdot/Z = -scale_rate (time to contact 1/scale_rate), independent of the object's true size, the camera pitch
// or the road slope.
//
//   per pass (light frame, any view; boxes are in full-frame pixels, patches come from the full-resolution
//   luma plane of the ring frame, so a wide-view box and a zoom-view box of the same object compare directly)
//     -> at most max_boxes boxes (largest first) get a grey patch of the inner part of the box (inner_frac of
//        width and height), at most patch_px on the longer side (area-averaged when downscaled, native pixels
//        when smaller), stored with the pass time
//     -> each box continues the chain whose predicted box (constant velocity of centre and log size) overlaps
//        it best (IoU >= match_iou, same class group); no tracker, chains end after chain_max_gap_s
//     -> the chain's oldest patch within [min_dt_s, max_dt_s] is the template; the current frame is resampled
//        around the box at the box-ratio scale prior, a normalised cross-correlation grid over log scale and
//        integer shift finds the peak, a Gauss-Newton (Lucas-Kanade) fit of log scale, shift, gain and offset
//        refines it; its covariance (residual variance x inverse normal matrix, inflated by the residual's
//        lag-1 autocorrelation) is the stated uncertainty
//     -> no value when the patch is tiny or textureless, a region leaves the frame, the box is clipped at the
//        frame edge, the match is poor (NCC, fit R^2), the fit leaves the search range or does not converge,
//        or the stated uncertainty is above max_std
#pragma once

#include <cstdint>
#include <vector>

#include "yolo.h"

namespace roadsense {
namespace signs {

struct RoadUserDet;

struct ScaleConfig {
    int max_boxes = 12;         // boxes measured per pass (largest area first)
    int patch_px = 24;          // template's longer side, patch pixels (native frame pixels for smaller boxes)
    int min_side_px = 8;        // inner region shorter side, frame pixels: below it no value
    float inner_frac = 0.8f;    // central part of the box used as the template
    double min_dt_s = 0.2, max_dt_s = 1.0;  // baseline window; the oldest patch inside it is used
    double chain_max_gap_s = 1.2;           // a chain without a box for this long ends
    float match_iou = 0.25f;    // box vs the chain's predicted box
    float min_ncc = 0.7f;       // grid peak
    float min_r2 = 0.6f;        // fit: 1 - residual variance / template variance
    float min_tex = 2.5f;       // template standard deviation, grey levels
    double max_std = 0.06;      // stated 1-sigma of the log scale change over the baseline: above it no value
    // Several baselines (signd --lights-ru-scale-multi N; 1 = the oldest patch only, as before): the current frame is
    // aligned with up to N earlier patches of the window (the oldest and others spread over it); the rate is the
    // generalised least-squares slope of their log scale changes over their baselines, the errors taken as
    // correlated by `multi_rho` (all share the current frame's noise). A baseline that disagrees with the others by
    // more than `multi_out` sigma is left out; the stated std grows with the scatter the baselines show.
    int baselines = 1;
    double multi_rho = 0.5, multi_out = 3.5;
    // Lamp pair (signd --lights-ru-scale-lamps; needs the lamp pairs of RoadUserDet::lamps, i.e. the view cues): the
    // rate of ln(lamp separation) over the chain's passes of the window, a weighted line fit; its std from the
    // centroids' stated errors, grown by the fit's own scatter. Combined with the patch value when both exist.
    bool lamps = false;
    bool patch = true;           // false: lamp pair only (evaluation of the lamp measurement by itself)
    double lamp_jump = 0.25;     // a step of ln(separation) between two passes above this: another pair, series ends
    double lamp_min_sep = 10.0;  // px: below, the pair picked flips between lamps and reflections (measured: 0.8 px)
    int lamp_min_n = 3;          // passes with the pair in the window (the fit's own scatter needs a third point)
    double lamp_huber = 1.5;     // the line fit down-weights a pass further than this many sigma from it
    double lamp_max_std = 0.12;  // stated 1-sigma of the lamp rate (1/s) above which it is not used
    double agree_sigma = 3.0;    // patch and lamp values further apart than this (in their combined sigma) disagree:
                                 // the mean is given with half their difference added to its std
};

// Why a box got no value (or kMeasured), counted per box by ScaleRateEstimator::why().
enum ScaleWhy {
    kMeasured = 0, kSmall, kTexture, kFrameEdge, kNcc, kSearchEdge, kFit, kRunaway, kNoConv, kR2, kStd,  // measure_scale()
    kOverCap, kBoxEdge, kNew, kNoBaseline, kClipped,                                  // the estimator
    kLampOnly, kNumScaleWhy  // measured by the lamp pair where the patch gave no value (counted beside the patch's reason)
};
// Where a box's value comes from (RoadUserDet::scale_src): patch alignment, the lamp pair, or both combined.
enum ScaleSrc { kSrcNone = 0, kSrcPatch = 1, kSrcLamps = 2, kSrcBoth = 3 };
const char *scale_why_name(int why);

struct ScaleResult {
    bool ok = false;
    int why = kSmall;
    double ln_s = 0, ln_s_std = 0, dt_s = 0;  // log scale change over the baseline, its 1-sigma, the baseline
    float ncc = 0, r2 = 0;
};

// The core measurement, exposed for tests: log scale change of the template patch `tpl` (tw x th, centred at
// frame point (tcx, tcy), k0 patch pixels per frame pixel, taken at the earlier frame) in the current frame `f`,
// with the scale prior `prior_ln_s` and the predicted centre (pcx, pcy) of the same object point in `f`.
// half_range: +- log-scale search range around the prior; shift_r: +- shift search range (search-image pixels).
ScaleResult measure_scale(const std::vector<float> &tpl, int tw, int th, double k0, const Nv12View &f, double prior_ln_s,
                          double pcx, double pcy, double half_range, int shift_r, const ScaleConfig &cfg);

// Grey patch of the frame's luma: centre (cx, cy) frame px, k patch px per frame px, w x h patch px; area
// averaging when k < 1. False when the patch would leave the frame.
bool sample_luma(const Nv12View &f, double cx, double cy, double k, int w, int h, std::vector<float> &out);

class ScaleRateEstimator {
public:
    explicit ScaleRateEstimator(const ScaleConfig &cfg) : cfg_(cfg) {}
    // One pass: measures the boxes of `users` (frame pixels of `f`) in place (RoadUserDet::scale_*), stores
    // their patches. ts_ns: the frame's monotonic time. `low` (optional): the low-score boxes; they continue the
    // chain of an object the first list missed in this pass (and are measured), and never start a chain.
    void update(const Nv12View &f, int64_t ts_ns, std::vector<RoadUserDet> &users, std::vector<RoadUserDet> *low = nullptr);
    void reset() { chains_.clear(); }
    size_t chains() const { return chains_.size(); }
    const ScaleConfig &config() const { return cfg_; }
    // boxes per ScaleWhy since construction
    const uint64_t *why() const { return why_; }

private:
    struct Entry {
        int64_t ts;
        float box[4];
        double cx, cy, k;  // template centre (frame px), patch px per frame px
        int w, h;
        bool edge;         // box touches the frame edge (no template)
        int clip;          // RoadUserDet::clip (edges of the view's region the box touches)
        bool lamp = false; // the box had a lamp pair
        float lsep = 0, lsep_std = 0;
        int lcol = 0;
        std::vector<float> tpl;
    };
    struct Chain {
        int grp;
        std::vector<Entry> e;  // oldest first
    };
    ScaleConfig cfg_;
    std::vector<Chain> chains_;
    int64_t last_ts_ = 0;
    uint64_t why_[kNumScaleWhy] = {};
};

}  // namespace signs
}  // namespace roadsense
