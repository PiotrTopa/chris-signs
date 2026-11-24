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

#include "detin.h"
#include "pose.h"
#include "runner.h"
#include "yolo.h"

namespace roadsense {
namespace signs {

class ScaleRateEstimator;

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
    // Road users (roadsense lane L42, signd --lights-road-users): boxes of the BDD100K classes 0..7 (person,
    // rider, car, truck, bus, train, motor, bike) from the same detector output, decoded only when this is on;
    // a rider is paired with the bike / motorbike under it into one road user. No extra inference: the cost is
    // the decode (road_users()). Non-critical (display metadata), shed first by the governor (`road_users_hz`).
    bool road_users = false;
    float ru_thr[8] = {0.40f, 0.35f, 0.40f, 0.40f, 0.40f, 0.50f, 0.35f, 0.35f};  // per class
    float ru_nms = 0.5f;           // class-agnostic NMS (one box, one class)
    int ru_max = 40;
    // Optional road-user view (view 2, signd --lights-ru-roi; off unless x1 > x0): an EXTRA detector run on this
    // part of the frame, road users only, at most ru_hz a second, after the light frame is published (the lights
    // never wait for it). The light views end at the horizon, so near road users there have no ground point; a
    // lower band gives them one. It costs one detector run each time: off by default, shed first.
    float ru_roi[4] = {0.f, 0.f, 0.f, 0.f};
    double ru_hz = 1.0;
    // Image scale change per road user (signd --lights-ru-scale, signd/scale.h): `scale_rate` (d ln size / dt of
    // the object's image, 1/s), `scale_rate_std`, `scale_dt_s` on the boxes that get a value. Needs road_users.
    bool ru_scale = false;
    int ru_scale_max = 12;  // boxes measured per pass (largest first)
    float ru_scale_dt[2] = {0.2f, 1.0f};  // baseline window (s): the oldest patch of the object inside it is used
    // Refinements of the scale rate (signd/scale.h), off by default: ru_scale_multi > 1 = that many baselines per
    // box (--lights-ru-scale-multi N); ru_scale_lamps = the lamp pair's separation as a second measurement in the
    // dark (--lights-ru-scale-lamps; switches the lamp search of ru_view on for the estimator even when the view
    // cues are not published). With ru_thr_low the low-score boxes continue an object's chain.
    int ru_scale_multi = 1;
    bool ru_scale_lamps = false;
    bool ru_scale_patch = true;  // false: lamp pair only (evaluation)
    // Detection inputs for a tracker (signd/detin.h, docs/DETECTION_INPUTS.md); all off by default, each adds only
    // optional keys. They need road_users.
    //   ru_thr_low > 0 (signd --lights-ru-low X): boxes scoring above X but not above their class threshold are
    //     listed too, in a list of their own (`road_users_low`): a tracker offers them to existing tracks only and
    //     never starts one from them. A low box that overlaps a listed road user by more than ru_nms is dropped.
    //   ru_flags (--lights-ru-flags): `trunc`, `occ`, `occ_by`, `occ_edges` per box.
    //   ru_view (--lights-ru-view): view cues of vehicles from the box's own pixels, no model: `sym` (mirror
    //     symmetry), `lamps` (the lamp pair of a vehicle in the dark: centroids, separation, colour).
    //   ru_emb (--lights-ru-emb yuv | PATH.onnx): an appearance vector per box (`emb`, hex of int8) and the message
    //     key `emb_kind`: "yuv" = the model-free yuv3x3 vector; a file = the learnt 32-number embedding followed by
    //     the yuv3x3 numbers (ORT, one run per box, at most ru_emb_max boxes a pass, largest first; the others
    //     carry none).
    std::string ru_emb;
    int ru_emb_max = 16;
    //   ru_pose (--lights-ru-pose PATH.onnx): heading and ground contact points per box from a trained head
    //     (signd/pose.h, docs/POSE_HEAD.md): `pose` per box and the message key `pose_kind`. One ORT run per box, at
    //     most ru_pose_max boxes a pass (first list by size, then the low list; with more boxes than that the passes
    //     take turns through the list); vehicles only unless ru_pose_all. ru_pose_cam = the camera's focal length and principal point in frame px (f, cu, cv; f 0:
    //     not known), given to the head. The model file is not part of this repository; a missing file = off.
    std::string ru_pose;
    int ru_pose_max = 4;
    bool ru_pose_all = false;
    float ru_pose_cam[3] = {0.f, 0.f, 0.f};
    float ru_thr_low = 0.f;
    int ru_low_max = 20;
    bool ru_flags = false;
    bool ru_view = false;
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

// One road user of a light frame (frame pixels). cls: 0 person, 1 rider (no vehicle found under it), 2 car,
// 3 truck, 4 bus, 5 train, 6 motor, 7 bike; a paired rider makes its vehicle's box the union and sets `rider`.
// clip: the box touches the edge of the view's region (bit 1 left, 2 top, 4 right, 8 bottom): a box cut at the
// bottom gives no ground contact point. p: the eight class scores of the box (for the consumer's vote).
struct RoadUserDet {
    float x0, y0, x1, y1;
    float score;
    int cls;
    int clip = 0;
    bool rider = false;
    float p[8];
    // Image scale change (LightsConfig::ru_scale, signd/scale.h): d ln(size)/dt of the object's image over the
    // baseline scale_dt_s (positive = growing = approaching), its 1-sigma; only when scale_ok.
    bool scale_ok = false;
    float scale_rate = 0.f, scale_rate_std = 0.f, scale_dt_s = 0.f;
    int scale_src = 0;  // ScaleSrc (signd/scale.h): 1 patch alignment, 2 lamp pair, 3 both; reported only with the lamps on
    // Flags (LightsConfig::ru_flags, signd/detin.h), valid when `flags`: trunc = frame edges that cut the box (bits
    // as clip); occ = share of the box covered by nearer boxes of the same message (0..1), occ_by = index of the
    // one that covers most (-1: none), occ_edges = edges of this box hidden by them (bits as clip).
    bool flags = false;
    int trunc = 0;
    float occ = 0.f;
    int occ_by = -1, occ_edges = 0;
    // View cues (LightsConfig::ru_view), valid when `view`: sym = mirror symmetry of the box's luma (-1..1; < -1:
    // not measured), lamps = the lamp pair (lamps.ok).
    bool view = false;
    float sym = -2.f;
    LampPair lamps;
    Appearance emb;  // LightsConfig::ru_emb; emb.n = 0: none
    RuPose pose;     // LightsConfig::ru_pose; pose.ok false: none
};
const char *road_user_name(int cls);

struct LightsTiming {
    double pre_ms = 0, det_ms = 0, post_ms = 0, state_ms = 0, total_ms = 0, wait_ms = 0, users_ms = 0;
    double scale_ms = -1;  // measure_scale() of this pass (< 0: not run)
    double flags_ms = 0, view_ms = 0, emb_ms = 0;  // parts of users_ms (detection inputs, signd/detin.h)
    double pose_ms = 0;                            // ... the pose head (signd/pose.h)
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
    bool has_ru_view() const { return cfg_.road_users && cfg_.ru_roi[2] > cfg_.ru_roi[0] && cfg_.ru_roi[3] > cfg_.ru_roi[1]; }
    // view 2 = the road-user view (process() then decodes road users only: no lights, no sign proposals)
    const float *view_roi(int view) const {
        return view == 2 && has_ru_view() ? cfg_.ru_roi : view == 1 && has_view2() ? cfg_.roi2 : cfg_.roi;
    }
    const LightsConfig &config() const { return cfg_; }
    const char *backend() const { return det_->name(); }
    bool has_state_cnn() const { return bool(st_); }
    // "" (off), "yuv3x3" or "reid32+yuv3x3": the kind of the road users' appearance vectors (message key `emb_kind`)
    const char *emb_kind() const { return cfg_.ru_emb.empty() ? "" : emb_ ? "reid32+yuv3x3" : "yuv3x3"; }
    // "" (off or no model file) or "pose1": the contract of the road users' `pose` (message key `pose_kind`)
    const char *pose_kind() const { return pose_ ? "pose1" : ""; }
    // The sign proposals of the last process() call (empty unless cfg.sign_cls >= 0), frame pixels, NMS'd.
    const std::vector<LightDet> &sign_boxes() const { return signs_; }
    // The road users of the last process() call (empty unless cfg.road_users and process() was asked to decode
    // them), frame pixels, NMS'd, riders paired.
    const std::vector<RoadUserDet> &road_users() const { return users_; }
    // The low-score road users of the last process() call (cfg.ru_thr_low > 0): boxes below their class threshold
    // that no listed road user suppresses; for a tracker's second association stage only.
    const std::vector<RoadUserDet> &road_users_low() const { return users_low_; }
    // Decode road users in the next process() calls (the governor's `road_users_hz` pauses this); default on
    // when cfg.road_users.
    void set_road_users(bool on) { ru_on_ = on && cfg_.road_users; }
    // Image scale change of the road users of the last process() call (cfg.ru_scale): fills their scale_* fields
    // from the full-resolution luma of `frame` (the frame process() just saw) and keeps their patches for later
    // passes. ts_ns: the frame's monotonic time. Adds its time to t->scale_ms and t->total_ms when given.
    void measure_scale(const Nv12View &frame, int64_t ts_ns, LightsTiming *t = nullptr);
    void reset_scale();
    // boxes per ScaleWhy (signd/scale.h) since start, nullptr without cfg.ru_scale
    const uint64_t *scale_why() const;
    bool road_users_on() const { return ru_on_; }
    // GPU schedule hooks, as SignPipeline::before_det / after_det.
    std::function<void()> before_det, after_det;

private:
    void classify(const Nv12View &f, LightDet &d);
    LightsConfig cfg_;
    std::unique_ptr<ModelRunner> det_, st_, emb_, pose_;
    std::vector<float> emb_in_, emb_out_, pose_in_, pose_out_;
    void embed(const Nv12View &f);
    void pose(const Nv12View &f);
    size_t pose_rr_ = 0;  // where the next pass starts in the list when there are more boxes than ru_pose_max
    int n_anchor_ = 0, lb_w_ = -1, lb_h_ = -1;
    std::vector<float> in_, out_, st_in_, st_out_;
    std::vector<float> rgb_;
    std::vector<LightDet> signs_;
    std::vector<RoadUserDet> users_, users_low_;
    bool ru_on_ = false;
    std::unique_ptr<ScaleRateEstimator> scale_;
};

// Pairs riders with the bike / motorbike box under them (in place): the vehicle keeps its class, gets the
// union box, `rider` and the higher score; the rider box goes. Exposed for tests.
void pair_riders(std::vector<RoadUserDet> &v);

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

// The road-user view's message (view 2): {"type":"road_users","seq","ts_mono_ns","ts_real_ns","ms","frame","roi","view":2,
// "road_users":[...]} on the light socket.
// low (optional): the second, low-score list (key "road_users_low", left out when nullptr).
std::string road_users_json(uint64_t seq, int64_t ts_mono_ns, int64_t ts_real_ns, double ms, int frame_w, int frame_h,
                            const float *roi, const std::vector<RoadUserDet> &users,
                            const std::vector<RoadUserDet> *low = nullptr, const char *emb_kind = nullptr,
                            const char *pose_kind = nullptr);

// users: the frame's road users (key "road_users"), or nullptr; users_state: 0 = the key is left out
// (--lights-road-users off), 1 = list, 2 = shed by the governor ("road_users": null). users_low (optional, only
// with users_state 1): the low-score list, key "road_users_low".
std::string lights_json(uint64_t seq, int64_t ts_mono_ns, int64_t ts_real_ns, double lights_ms, int frame_w,
                        int frame_h, const LightsConfig &cfg, const std::vector<LightTrackReport> &tracks,
                        const std::vector<LightDet> *raw = nullptr, const LightsTiming *timing = nullptr, int view = 0,
                        const std::vector<RoadUserDet> *users = nullptr, int users_state = 0,
                        const std::vector<RoadUserDet> *users_low = nullptr, const char *emb_kind = nullptr,
                        const char *pose_kind = nullptr);
// One road user as a JSON object (the element of "road_users" / "road_users_low").
std::string road_user_json(const RoadUserDet &d);

// Idle pattern (--lights-idle): the passes a second of `pattern` at `period_hz` frame periods a second (letters
// other than '.' are passes), and the message key a consumer reads to judge the stream's rate while the light
// detector tracks nothing: `,"idle":{"pattern":"ZW.Z.W","hz":3.33}` inserted before the message's closing brace.
// Messages without the key come from the full schedule (a light is tracked, the pattern is off or forbidden).
double lights_idle_hz(const std::string &pattern, double period_hz);
void add_lights_idle(std::string &msg, const std::string &pattern, double hz);

}  // namespace signs
}  // namespace roadsense
