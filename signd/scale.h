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
};

// Why a box got no value (or kMeasured), counted per box by ScaleRateEstimator::why().
enum ScaleWhy {
    kMeasured = 0, kSmall, kTexture, kFrameEdge, kNcc, kSearchEdge, kFit, kRunaway, kNoConv, kR2, kStd,  // measure_scale()
    kOverCap, kBoxEdge, kNew, kNoBaseline, kClipped, kNumScaleWhy                     // the estimator
};
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
    // their patches. ts_ns: the frame's monotonic time.
    void update(const Nv12View &f, int64_t ts_ns, std::vector<RoadUserDet> &users);
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
