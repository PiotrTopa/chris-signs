// Traffic lights for signd (roadsense docs/ARCHITECTURE.md, `lights` topic; measurements and model choice:
// roadsense docs/lanes/L7-traffic-lights.md, models: docs/LIGHT_MODELS.md).
//
//   NV12 frame
//     -> region of interest (default: the upper part of the frame, where lights are) resampled to the
//        detector input, content at the top-left, grey 114 padding (same resampler as the sign letterbox)
//     -> BDD100K YOLO11n detector (10 road classes, Ultralytics head [4+10, N]); only class 8
//        "traffic light" is kept, score > thr, greedy NMS, mapped back to frame pixels
//     -> state per box from the NV12 crop: a tiny CNN (red / yellow / green / none, optional) and/or
//        the lamp colour rule (hue + brightness + lamp position along the housing); red+yellow from the
//        rule (two lit lamps), flashing from the track history
//     -> LightTracker: greedy IoU / centre association, smoothed state per track
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "runner.h"
#include "yolo.h"

namespace roadsense {
namespace signs {

enum LightState { kRed = 0, kYellow, kRedYellow, kGreen, kOff, kUnknown, kNumLightStates };
const char *light_state_name(int s);

struct LightsConfig {
    std::string backend = "ncnn";  // ncnn (Vulkan), ncnn-cpu, ort
    std::string models_dir = "/var/lib/roadsense/models";
    std::string model;             // default: models_dir/bdd100k_yolo11n_<H>x<W>{-ncnn.param,.onnx}
    int in_w = 960, in_h = 320;     // detector input (W x H); the model file is named H x W
    float roi[4] = {0.f, 0.f, 1.f, 0.62f};  // x0, y0, x1, y1 as fractions of the frame
    // Second view (optional, x1 > x0 enables it): a smaller part of the frame fed to the same detector at a
    // higher scale, on every other processed frame. The wide view finds the near lights anywhere above the
    // road, the second one the far lights around the vanishing point that are too small after the wide
    // view's downscale (roadsense docs/lanes/L15-stability.md).
    float roi2[4] = {0.f, 0.f, 0.f, 0.f};
    int n_cls = 10, cls_id = 8;    // BDD100K: 8 = traffic light
    // Sign proposals (roadsense lane L24-S, signd --lights-signs): boxes of this class (BDD100K 9 = traffic sign)
    // scoring > sign_thr are kept too (sign_boxes()), for the sign classifier; -1 = off.
    int sign_cls = -1;
    float sign_thr = 0.35f;
    float det_thr = 0.25f, nms_iou = 0.45f;
    // Boxes down to this score are kept for the tracker, which lets them continue an existing track but never
    // start one (0 = same as det_thr: every box may start a track, the behaviour before 2026-11-16).
    float det_thr_keep = 0.f;
    int max_det = 30;
    float min_h_px = 6.f;          // drop boxes shorter than this (frame px)
    int threads = 2;               // ORT intra-op / ncnn CPU threads
    bool ncnn_fp16 = true, ncnn_flushall = true;
    // State: "rule" (colour rule only) or a path to the state CNN (.onnx on ORT, 1x3x48x24 -> 4 logits
    // red, yellow, green, none), which the rule then refines (red+yellow, off).
    std::string state = "rule";
    float rule_v_min = 0.55f, rule_s_min = 0.30f, rule_v_rel = 0.70f, rule_min = 0.0075f, rule_second = 0.35f;
};

struct LightDet {
    float x0, y0, x1, y1;  // frame pixels
    float score;
    int state = kUnknown;  // this frame's state
    float state_conf = 0;  // its probability (CNN) or rule strength (0..1)
    float rule[3] = {0, 0, 0};  // rule scores red, yellow, green (lit-lamp evidence)
    bool horizontal = false;
};

struct LightsTiming {
    double pre_ms = 0, det_ms = 0, post_ms = 0, state_ms = 0, total_ms = 0, wait_ms = 0;
};

class LightPipeline {
public:
    explicit LightPipeline(const LightsConfig &cfg);  // throws std::runtime_error
    ~LightPipeline();
    LightPipeline(const LightPipeline &) = delete;
    LightPipeline &operator=(const LightPipeline &) = delete;

    // view 0 = cfg.roi, view 1 = cfg.roi2 (falls back to roi when no second view is configured).
    void process(const Nv12View &frame, std::vector<LightDet> &out, LightsTiming *t = nullptr, int view = 0);
    bool has_view2() const { return cfg_.roi2[2] > cfg_.roi2[0] && cfg_.roi2[3] > cfg_.roi2[1]; }
    const float *view_roi(int view) const { return view == 1 && has_view2() ? cfg_.roi2 : cfg_.roi; }
    const LightsConfig &config() const { return cfg_; }
    const char *backend() const { return det_->name(); }
    bool has_state_cnn() const { return bool(st_); }
    // The sign proposals of the last process() call (empty unless cfg.sign_cls >= 0), frame pixels, NMS'd.
    const std::vector<LightDet> &sign_boxes() const { return signs_; }
    // GPU schedule hooks, as SignPipeline::before_det / after_det.
    std::function<void()> before_det, after_det;

private:
    void classify(const Nv12View &f, LightDet &d);
    LightsConfig cfg_;
    std::unique_ptr<ModelRunner> det_, st_;
    int n_anchor_ = 0, lb_w_ = -1, lb_h_ = -1;
    std::vector<float> in_, out_, st_in_, st_out_;
    std::vector<float> rgb_;
    std::vector<LightDet> signs_;
};

// Colour rule on an RGB crop (planar float 0..1, w x h), vertical housing assumed unless horizontal:
// lamp evidence per colour (red top, yellow middle, green bottom). Exposed for tests/tools.
void light_rule_scores(const float *rgb, int w, int h, bool horizontal, const LightsConfig &cfg, float out[3]);

struct LightTrackerConfig {
    int max_age = 6;          // processed frames without a match before a track ends
    int min_hits = 2;         // hits before a track is reported
    float iou_match = 0.1f;
    float dist_match = 1.5f;  // centre distance / max(w,h) for the fallback pass
    double flash_window_s = 4.0;  // flashing yellow: yellow <-> off alternation inside this window
    // A box scoring below this continues a track but does not start one (LightsConfig::det_thr_keep lets
    // such boxes through); 0 = every box may start a track.
    float new_score = 0.f;
    float dist_match_weak = 0.75f;  // fallback association distance for those weak boxes
    // A confirmed track that was not matched in this frame is still reported for this many processed
    // frames, at its predicted box and with its last state (`coast` = frames since the last match): a
    // light the detector misses in one frame, or one only the other view can see, does not vanish from the
    // message. 0 = matched tracks only (the behaviour before 2026-11-16).
    int report_coast = 0;
};

struct LightTrackReport {
    int id;
    float box[4];
    float score;
    int state;        // smoothed
    float state_conf;
    int frame_state;  // this frame's observation (kUnknown if not matched this frame)
    int coast = 0;    // processed frames since the last match (0 = matched in this frame)
    bool flashing;
    bool horizontal;
    int hits, age;
    double state_s;   // time since the smoothed state last changed
};

class LightTracker {
public:
    explicit LightTracker(const LightTrackerConfig &cfg) : cfg_(cfg) {}
    // `seen` (optional): the part of the frame the detector looked at this time, x0, y0, x1, y1 in frame
    // pixels; a track whose centre lies outside it could not have been matched and does not age.
    void update(uint64_t seq, int64_t ts_ns, const std::vector<LightDet> &dets, std::vector<LightTrackReport> &out,
                const float *seen = nullptr);
    void reset() { tracks_.clear(); }
    size_t live() const { return tracks_.size(); }

private:
    struct Obs {
        int64_t ts;
        int state;
        float conf;
    };
    struct Track {
        int id;
        float box[4], v[2] = {0, 0};
        float score = 0;
        int hits = 0, age = 0;
        int since = 0;  // observable frames without a match (ends the track above max_age)
        int steps = 0;  // processed frames since the last match (the reported `coast`)
        int state = kUnknown, cand = kUnknown, cand_n = 0;
        float state_conf = 0;
        int64_t state_t = 0;
        bool horizontal = false;
        std::vector<Obs> hist;  // recent observations (newest last)
    };
    void observe(Track &t, int64_t ts, const LightDet &d);
    bool flashing(const Track &t, int64_t ts) const;
    LightTrackerConfig cfg_;
    std::vector<Track> tracks_;
    int next_id_ = 1;
};

std::string lights_json(uint64_t seq, int64_t ts_mono_ns, int64_t ts_real_ns, double lights_ms, int frame_w,
                        int frame_h, const LightsConfig &cfg, const std::vector<LightTrackReport> &tracks,
                        const std::vector<LightDet> *raw = nullptr, const LightsTiming *timing = nullptr, int view = 0);

}  // namespace signs
}  // namespace roadsense
