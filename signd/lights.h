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
    int n_cls = 10, cls_id = 8;    // BDD100K: 8 = traffic light
    float det_thr = 0.25f, nms_iou = 0.45f;
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

    void process(const Nv12View &frame, std::vector<LightDet> &out, LightsTiming *t = nullptr);
    const LightsConfig &config() const { return cfg_; }
    const char *backend() const { return det_->name(); }
    bool has_state_cnn() const { return bool(st_); }
    // GPU schedule hooks, as SignPipeline::before_det / after_det.
    std::function<void()> before_det, after_det;

private:
    void classify(const Nv12View &f, LightDet &d);
    LightsConfig cfg_;
    std::unique_ptr<ModelRunner> det_, st_;
    int n_anchor_ = 0, lb_w_ = -1, lb_h_ = -1;
    std::vector<float> in_, out_, st_in_, st_out_;
    std::vector<float> rgb_;
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
};

struct LightTrackReport {
    int id;
    float box[4];
    float score;
    int state;        // smoothed
    float state_conf;
    int frame_state;  // this frame's observation (kUnknown if not matched this frame)
    bool flashing;
    bool horizontal;
    int hits, age;
    double state_s;   // time since the smoothed state last changed
};

class LightTracker {
public:
    explicit LightTracker(const LightTrackerConfig &cfg) : cfg_(cfg) {}
    void update(uint64_t seq, int64_t ts_ns, const std::vector<LightDet> &dets, std::vector<LightTrackReport> &out);
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
        int hits = 0, age = 0, since = 0;
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
                        const std::vector<LightDet> *raw = nullptr, const LightsTiming *timing = nullptr);

}  // namespace signs
}  // namespace roadsense
