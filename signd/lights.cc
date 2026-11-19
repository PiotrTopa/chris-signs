#include "lights.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <unistd.h>

namespace roadsense {
namespace signs {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

float iou4(const float *a, const float *b) {
    const float ix = std::max(0.f, std::min(a[2], b[2]) - std::max(a[0], b[0]));
    const float iy = std::max(0.f, std::min(a[3], b[3]) - std::max(a[1], b[1]));
    const float i = ix * iy;
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9f);
}

std::string fmt(double v, int prec) {
    char b[48];
    snprintf(b, sizeof(b), "%.*f", prec, v);
    return b;
}

constexpr int kStH = 48, kStW = 24;  // state CNN input (H x W)
}  // namespace

const char *road_user_name(int cls) {
    static const char *n[] = {"person", "rider", "car", "truck", "bus", "train", "motor", "bike"};
    return cls >= 0 && cls < 8 ? n[cls] : "unknown";
}

void pair_riders(std::vector<RoadUserDet> &v) {
    // A rider stands on its vehicle: the two overlap horizontally by most of the narrower box, the rider's
    // box reaches into the vehicle's and its centre is above the vehicle's. Best overlap first, one rider per
    // vehicle.
    std::vector<char> gone(v.size(), 0), taken(v.size(), 0);
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i].cls != 1) continue;
        const RoadUserDet &r = v[i];
        int best = -1;
        float best_ov = 0.f;
        for (size_t j = 0; j < v.size(); ++j) {
            if ((v[j].cls != 6 && v[j].cls != 7) || taken[j]) continue;
            const RoadUserDet &b = v[j];
            const float ox = std::min(r.x1, b.x1) - std::max(r.x0, b.x0);
            const float ov = ox / std::max(1.f, std::min(r.x1 - r.x0, b.x1 - b.x0));
            if (ov < 0.5f || r.y1 < b.y0 || (r.y0 + r.y1) > (b.y0 + b.y1) || ov <= best_ov) continue;
            best = int(j), best_ov = ov;
        }
        if (best < 0) continue;
        RoadUserDet &b = v[size_t(best)];
        b.x0 = std::min(b.x0, r.x0), b.y0 = std::min(b.y0, r.y0), b.x1 = std::max(b.x1, r.x1), b.y1 = std::max(b.y1, r.y1);
        b.score = std::max(b.score, r.score);
        b.clip |= r.clip;
        b.rider = true;
        taken[size_t(best)] = 1, gone[i] = 1;
    }
    size_t k = 0;
    for (size_t i = 0; i < v.size(); ++i)
        if (!gone[i]) v[k++] = v[i];
    v.resize(k);
}

const char *light_state_name(int s) {
    static const char *n[] = {"red", "yellow", "red_yellow", "green", "off", "unknown"};
    return s >= 0 && s < kNumLightStates ? n[s] : "unknown";
}

// ------------------------------------------------------------------ rule ---
// Per pixel: HSV value v = max(r,g,b), saturation s = (max-min)/max, hue in degrees. A pixel is "lit"
// if v >= max(v_min, v_rel * p99(v of the crop)) and s >= s_min. Its weight s*v goes to the colour
// whose hue band it falls in (red 345..20, yellow 20..60, green 130..210 deg; LED greens are cyan-ish),
// times a triangular position prior along the housing (red lamp centre at 1/6, yellow 1/2, green 5/6,
// zero beyond 2/3 of the housing away). Scores are normalised by the crop area.
void light_rule_scores(const float *rgb, int w, int h, bool horizontal, const LightsConfig &cfg, float out[3]) {
    out[0] = out[1] = out[2] = 0;
    const size_t n = size_t(w) * h;
    if (w < 2 || h < 3) return;
    const float *R = rgb, *G = rgb + n, *B = rgb + 2 * n;
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = std::max(R[i], std::max(G[i], B[i]));
    std::vector<float> vs(v);
    const size_t k = std::min(n - 1, size_t(std::floor(0.99 * double(n - 1))));
    std::nth_element(vs.begin(), vs.begin() + long(k), vs.end());
    const float thr = std::max(cfg.rule_v_min, cfg.rule_v_rel * vs[k]);
    static const float centre[3] = {1.f / 6, 0.5f, 5.f / 6};
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const size_t i = size_t(y) * w + x;
            const float mx = v[i];
            if (mx < thr) continue;
            const float r = R[i], g = G[i], b = B[i];
            const float mn = std::min(r, std::min(g, b)), d = mx - mn + 1e-6f;
            const float s = mx > 0 ? (mx - mn) / (mx + 1e-6f) : 0.f;
            if (s < cfg.rule_s_min) continue;
            float hue;
            if (mx == r) hue = std::fmod((g - b) / d + 6.f, 6.f) * 60.f;
            else if (mx == g) hue = ((b - r) / d + 2.f) * 60.f;
            else hue = ((r - g) / d + 4.f) * 60.f;
            int c = -1;
            if (hue >= 345.f || hue < 20.f) c = 0;
            else if (hue < 60.f) c = 1;
            else if (hue >= 130.f && hue < 210.f) c = 2;
            if (c < 0) continue;
            const float pos = horizontal ? (x + 0.5f) / w : (y + 0.5f) / h;
            const float pw = std::min(1.f, std::max(0.f, 1.f - std::fabs(pos - centre[c]) * 1.5f));
            out[c] += s * mx * pw;
        }
    for (int c = 0; c < 3; ++c) out[c] /= float(n);
}

// -------------------------------------------------------------- pipeline ---

LightPipeline::LightPipeline(const LightsConfig &cfg) : cfg_(cfg) {
    if (cfg_.model.empty()) {
        cfg_.model = cfg_.models_dir + "/bdd100k_yolo11n_" + std::to_string(cfg_.in_h) + "x" + std::to_string(cfg_.in_w) +
                     (cfg_.backend == "ort" ? ".onnx" : "-ncnn.param");
    }
    if (cfg_.in_w % 32 || cfg_.in_h % 32) throw std::runtime_error("lights: input size must be multiples of 32");
    n_anchor_ = 0;
    for (int s : {8, 16, 32}) n_anchor_ += (cfg_.in_w / s) * (cfg_.in_h / s);
    RunnerOptions ro;
    ro.backend = cfg_.backend;
    ro.model = cfg_.model;
    ro.threads = cfg_.threads;
    ro.ncnn_fp16 = cfg_.ncnn_fp16;
    ro.ncnn_flushall = cfg_.ncnn_flushall;
    ro.input_shape = {1, 3, cfg_.in_h, cfg_.in_w};
    ro.output_size = size_t(4 + cfg_.n_cls) * n_anchor_;
    det_ = make_runner(ro);
    in_.assign(size_t(3) * cfg_.in_w * cfg_.in_h, 114.f / 255.f);
    out_.assign(ro.output_size, 0.f);
    if (cfg_.state.find(',') != std::string::npos) {
        // A list of state models, best first: the first readable file is used (a newer model that a host
        // may not have installed yet, then the one before it).
        std::string pick;
        size_t a = 0;
        while (a <= cfg_.state.size()) {
            const size_t b = std::min(cfg_.state.find(',', a), cfg_.state.size());
            const std::string f = cfg_.state.substr(a, b - a);
            if (!f.empty() && (f == "rule" || access(f.c_str(), R_OK) == 0)) {
                pick = f;
                break;
            }
            if (!f.empty()) pick = f;  // none readable: report the last one below
            a = b + 1;
        }
        cfg_.state = pick;
    }
    if (cfg_.state != "rule" && access(cfg_.state.c_str(), R_OK) != 0) {
        // The state CNN is trained on research-licensed data and installed by hand (docs/LIGHT_MODELS.md):
        // without it the lamp colour rule decides.
        fprintf(stderr, "signd: lights: state model %s not found, using the colour rule\n", cfg_.state.c_str());
        cfg_.state = "rule";
    }
    ru_on_ = cfg_.road_users;
    if (cfg_.state != "rule") {
        RunnerOptions so;
        so.backend = "ort";
        so.model = cfg_.state;
        so.threads = 1;
        so.input_shape = {1, 3, kStH, kStW};
        so.output_size = 4;
        st_ = make_runner(so);
        st_in_.assign(size_t(3) * kStH * kStW, 0.f);
        st_out_.assign(4, 0.f);
    }
}

LightPipeline::~LightPipeline() = default;

void LightPipeline::classify(const Nv12View &f, LightDet &d) {
    const int bw = std::max(2, int(std::lround(d.x1 - d.x0))), bh = std::max(3, int(std::lround(d.y1 - d.y0)));
    d.horizontal = bw > 1.2f * bh;
    // Rule on the crop at (at most) native resolution, capped at 64 px along the housing.
    const double sc = std::min(1.0, 64.0 / std::max(bw, bh));
    const int cw = std::max(2, int(std::lround(bw * sc))), ch = std::max(3, int(std::lround(bh * sc)));
    rgb_.resize(size_t(3) * cw * ch);
    nv12_region_to_rgb_planar(f, d.x0, d.y0, d.x1 - d.x0, d.y1 - d.y0, cw, ch, rgb_.data(), cw, size_t(cw) * ch, 0, 0);
    light_rule_scores(rgb_.data(), cw, ch, d.horizontal, cfg_, d.rule);
    const float *r = d.rule;
    const int best = int(std::max_element(r, r + 3) - r);
    const float bs = r[best];
    int rule_state = kOff;
    if (bs >= cfg_.rule_min) {
        rule_state = best == 0 ? kRed : best == 1 ? kYellow : kGreen;
        if (best <= 1) {
            const float other = r[1 - best];
            if (other >= cfg_.rule_second * bs && r[2] < 0.5f * other) rule_state = kRedYellow;
        }
    }
    const float rule_conf = std::min(1.f, bs / (4 * cfg_.rule_min));
    if (!st_) {
        d.state = rule_state;
        d.state_conf = rule_state == kOff ? 1.f - rule_conf : rule_conf;
        return;
    }
    // CNN on a 24 x 48 (W x H) resample; horizontal housings rotated clockwise (left lamp on top).
    const size_t pl = size_t(kStW) * kStH;
    if (!d.horizontal) {
        nv12_region_to_rgb_planar(f, d.x0, d.y0, d.x1 - d.x0, d.y1 - d.y0, kStW, kStH, st_in_.data(), kStW, pl, 0, 0);
    } else {
        std::vector<float> tmp(size_t(3) * pl);  // W 48 x H 24 in the original orientation
        nv12_region_to_rgb_planar(f, d.x0, d.y0, d.x1 - d.x0, d.y1 - d.y0, kStH, kStW, tmp.data(), kStH, pl, 0, 0);
        for (int c = 0; c < 3; ++c)
            for (int i = 0; i < kStH; ++i)      // out row i  <- old column i
                for (int j = 0; j < kStW; ++j)  // out col j  <- old row (kStW-1-j)
                    st_in_[c * pl + size_t(i) * kStW + j] = tmp[c * pl + size_t(kStW - 1 - j) * kStH + i];
    }
    st_->run(st_in_.data(), st_out_.data());
    float m = *std::max_element(st_out_.begin(), st_out_.end()), p[4], z = 0;
    for (int i = 0; i < 4; ++i) z += p[i] = std::exp(st_out_[i] - m);
    for (int i = 0; i < 4; ++i) p[i] /= z;
    const int k = int(std::max_element(p, p + 4) - p);
    static const int map[4] = {kRed, kYellow, kGreen, kOff};
    d.state = map[k];
    d.state_conf = p[k];
    // Red+yellow is not in the CNN's classes (BDD100K, USA) and the colour rule cannot tell it from a red lamp's
    // glare (L7 report): the tracker infers it from the sequence instead (yellow right after red).
}

void LightPipeline::process(const Nv12View &f, std::vector<LightDet> &out, LightsTiming *tm, int view) {
    out.clear();
    LightsTiming t;
    const auto t0 = Clock::now();
    const float *roi = view_roi(view);
    const double rx0 = std::floor(roi[0] * f.width), ry0 = std::floor(roi[1] * f.height);
    const double rx1 = std::ceil(roi[2] * f.width), ry1 = std::ceil(roi[3] * f.height);
    const double rw = std::max(2.0, rx1 - rx0), rh = std::max(2.0, ry1 - ry0);
    const double r = std::min(cfg_.in_w / rw, cfg_.in_h / rh);
    const int nw = std::min(cfg_.in_w, int(std::lround(rw * r))), nh = std::min(cfg_.in_h, int(std::lround(rh * r)));
    const size_t plane = size_t(cfg_.in_w) * cfg_.in_h;
    if (nw != lb_w_ || nh != lb_h_) {  // the padding only changes with the view's letterbox
        std::fill(in_.begin(), in_.end(), 114.f / 255.f);
        lb_w_ = nw;
        lb_h_ = nh;
    }
    nv12_region_to_rgb_planar(f, rx0, ry0, rw, rh, nw, nh, in_.data(), cfg_.in_w, plane, 0, 0);
    const auto t1 = Clock::now();
    t.pre_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (before_det) {
        det_->run_hooked(in_.data(), out_.data(), [&] {
            const auto tw = Clock::now();
            before_det();
            t.wait_ms = ms_since(tw);
        });
    } else {
        det_->run(in_.data(), out_.data());
    }
    if (after_det) after_det();
    const auto t2 = Clock::now();
    t.det_ms = std::chrono::duration<double, std::milli>(t2 - t1).count() - t.wait_ms;

    const size_t N = size_t(n_anchor_);
    const float *o = out_.data(), *sc = o + size_t(4 + cfg_.cls_id) * N;
    std::vector<LightDet> cand;
    const float thr = cfg_.det_thr_keep > 0 ? std::min(cfg_.det_thr, cfg_.det_thr_keep) : cfg_.det_thr;
    const bool lights_view = view != 2;                  // the road-user view decodes road users only
    for (size_t i = 0; i < N && lights_view; ++i) {
        if (!(sc[i] > thr)) continue;
        const float cx = o[i], cy = o[N + i], w = o[2 * N + i], h = o[3 * N + i];
        LightDet d;
        d.x0 = float(rx0 + (cx - w / 2) / r);
        d.y0 = float(ry0 + (cy - h / 2) / r);
        d.x1 = float(rx0 + (cx + w / 2) / r);
        d.y1 = float(ry0 + (cy + h / 2) / r);
        d.x0 = std::max(d.x0, 0.f), d.y0 = std::max(d.y0, 0.f);
        d.x1 = std::min(d.x1, float(f.width)), d.y1 = std::min(d.y1, float(f.height));
        d.score = sc[i];
        if (d.y1 - d.y0 < cfg_.min_h_px || d.x1 - d.x0 < 2) continue;
        cand.push_back(d);
    }
    std::stable_sort(cand.begin(), cand.end(), [](const LightDet &a, const LightDet &b) { return a.score > b.score; });
    std::vector<char> dead(cand.size(), 0);
    for (size_t i = 0; i < cand.size() && int(out.size()) < cfg_.max_det; ++i) {
        if (dead[i]) continue;
        for (size_t j = i + 1; j < cand.size(); ++j)
            if (!dead[j] && iou4(&cand[i].x0, &cand[j].x0) > cfg_.nms_iou) dead[j] = 1;
        out.push_back(cand[i]);
    }
    signs_.clear();
    if (lights_view && cfg_.sign_cls >= 0 && cfg_.sign_cls < cfg_.n_cls) {  // sign proposals: same decode, own class row
        const float *ss = o + size_t(4 + cfg_.sign_cls) * N;
        std::vector<LightDet> sc;
        for (size_t i = 0; i < N; ++i) {
            if (!(ss[i] > cfg_.sign_thr)) continue;
            const float cx = o[i], cy = o[N + i], w = o[2 * N + i], h = o[3 * N + i];
            LightDet d;
            d.x0 = std::max(float(rx0 + (cx - w / 2) / r), 0.f);
            d.y0 = std::max(float(ry0 + (cy - h / 2) / r), 0.f);
            d.x1 = std::min(float(rx0 + (cx + w / 2) / r), float(f.width));
            d.y1 = std::min(float(ry0 + (cy + h / 2) / r), float(f.height));
            d.score = ss[i];
            if (d.x1 - d.x0 < 4 || d.y1 - d.y0 < 4) continue;
            sc.push_back(d);
        }
        std::stable_sort(sc.begin(), sc.end(), [](const LightDet &a, const LightDet &b) { return a.score > b.score; });
        std::vector<char> sd(sc.size(), 0);
        for (size_t i = 0; i < sc.size() && signs_.size() < 30; ++i) {
            if (sd[i]) continue;
            for (size_t j = i + 1; j < sc.size(); ++j)
                if (!sd[j] && iou4(&sc[i].x0, &sc[j].x0) > cfg_.nms_iou) sd[j] = 1;
            signs_.push_back(sc[i]);
        }
    }
    users_.clear();
    if (ru_on_ || (view == 2 && cfg_.road_users)) {  // road users: same decode, the eight road classes, one class per box
        const auto tu = Clock::now();
        std::vector<RoadUserDet> uc;
        float tmin = 1.f;
        for (int c = 0; c < 8; ++c) tmin = std::min(tmin, cfg_.ru_thr[c]);
        const float ex0 = float(rx0) + 2.f, ey0 = float(ry0) + 2.f, ex1 = float(std::min(rx1, double(f.width))) - 2.f,
                    ey1 = float(std::min(ry1, double(f.height))) - 2.f;
        for (size_t i = 0; i < N; ++i) {
            int bc = 0;
            float bs = o[4 * N + i];
            for (int c = 1; c < 8; ++c) {
                const float v = o[size_t(4 + c) * N + i];
                if (v > bs) bs = v, bc = c;
            }
            if (!(bs > tmin) || !(bs > cfg_.ru_thr[bc])) continue;
            const float cx = o[i], cy = o[N + i], w = o[2 * N + i], h = o[3 * N + i];
            RoadUserDet d;
            d.x0 = std::max(float(rx0 + (cx - w / 2) / r), 0.f);
            d.y0 = std::max(float(ry0 + (cy - h / 2) / r), 0.f);
            d.x1 = std::min(float(rx0 + (cx + w / 2) / r), float(f.width));
            d.y1 = std::min(float(ry0 + (cy + h / 2) / r), float(f.height));
            if (d.x1 - d.x0 < 3 || d.y1 - d.y0 < 3) continue;
            d.score = bs, d.cls = bc;
            for (int c = 0; c < 8; ++c) d.p[c] = o[size_t(4 + c) * N + i];
            d.clip = (d.x0 <= ex0 ? 1 : 0) | (d.y0 <= ey0 ? 2 : 0) | (d.x1 >= ex1 ? 4 : 0) | (d.y1 >= ey1 ? 8 : 0);
            uc.push_back(d);
        }
        std::stable_sort(uc.begin(), uc.end(), [](const RoadUserDet &a, const RoadUserDet &b) { return a.score > b.score; });
        std::vector<char> ud(uc.size(), 0);
        for (size_t i = 0; i < uc.size() && int(users_.size()) < cfg_.ru_max; ++i) {
            if (ud[i]) continue;
            for (size_t j = i + 1; j < uc.size(); ++j)
                if (!ud[j] && iou4(&uc[i].x0, &uc[j].x0) > cfg_.ru_nms) ud[j] = 1;
            users_.push_back(uc[i]);
        }
        pair_riders(users_);
        t.users_ms = ms_since(tu);
    }
    const auto t3 = Clock::now();
    t.post_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() - t.users_ms;
    for (LightDet &d : out) classify(f, d);
    t.state_ms = ms_since(t3);
    t.total_ms = ms_since(t0);
    if (tm) *tm = t;
}

// --------------------------------------------------------------- tracker ---

void LightTracker::observe(Track &t, int64_t ts, const LightDet &d) {
    t.hist.push_back({ts, d.state, d.state_conf});
    while (!t.hist.empty() && ts - t.hist.front().ts > int64_t(cfg_.flash_window_s * 1e9)) t.hist.erase(t.hist.begin());
    // Smoothed state: switches after two consecutive observations of a new state, or one confident one.
    if (d.state == t.state) {
        t.cand_n = 0;
        t.state_conf = 0.7f * t.state_conf + 0.3f * d.state_conf;
        return;
    }
    if (d.state == t.cand) ++t.cand_n;
    else t.cand = d.state, t.cand_n = 1;
    if (t.state == kUnknown || t.cand_n >= 2 || d.state_conf >= 0.9f) {
        // Vienna-convention sequence (PL, DE, ...): yellow straight after red is the red+yellow phase; plain yellow
        // follows green. A red observation in the first 2.5 s of red+yellow keeps it (the lit red lamp dominates
        // the crop; the phase lasts 1-2 s), later it is red again.
        int ns = d.state;
        if (ns == kYellow && (t.state == kRed || t.state == kRedYellow)) ns = kRedYellow;
        if (ns == kRed && t.state == kRedYellow && ts - t.state_t < int64_t(2.5e9)) {
            t.cand_n = 0;
            return;
        }
        if (ns == t.state) {
            t.cand_n = 0;
            return;
        }
        t.state = ns;
        t.state_conf = d.state_conf;
        t.state_t = ts;
        t.cand_n = 0;
    }
}

bool LightTracker::flashing(const Track &t, int64_t ts) const {
    // Yellow and dark observations alternating: at least two of each and three changes between them.
    int ny = 0, no = 0, changes = 0, last = -1;
    for (const Obs &o : t.hist) {
        if (ts - o.ts > int64_t(cfg_.flash_window_s * 1e9)) continue;
        const int s = o.state == kYellow ? 1 : o.state == kOff ? 0 : -1;
        if (s < 0) return false;
        ny += s == 1, no += s == 0;
        if (last >= 0 && s != last) ++changes;
        last = s;
    }
    return ny >= 2 && no >= 2 && changes >= 3;
}

void LightTracker::update(uint64_t, int64_t ts, const std::vector<LightDet> &dets, std::vector<LightTrackReport> &out,
                          const float *seen) {
    out.clear();
    for (Track &t : tracks_) {  // constant-velocity prediction of the box centre
        t.box[0] += t.v[0], t.box[2] += t.v[0], t.box[1] += t.v[1], t.box[3] += t.v[1];
        ++t.age, ++t.steps;
        // A track outside the part of the frame the detector looked at this time cannot be matched: it moves
        // on with its velocity and is not counted as missed.
        const float cx = (t.box[0] + t.box[2]) / 2, cy = (t.box[1] + t.box[3]) / 2;
        if (!seen || (cx >= seen[0] && cx <= seen[2] && cy >= seen[1] && cy <= seen[3])) ++t.since;
    }
    const auto weak = [&](const LightDet &d) { return cfg_.new_score > 0 && d.score < cfg_.new_score; };
    std::vector<int> match(dets.size(), -1);
    std::vector<char> used(tracks_.size(), 0);
    struct P { float s; size_t d, t; };
    std::vector<P> pairs;
    for (size_t i = 0; i < dets.size(); ++i)
        for (size_t j = 0; j < tracks_.size(); ++j) {
            const float u = iou4(&dets[i].x0, tracks_[j].box);
            if (u >= cfg_.iou_match) pairs.push_back({u, i, j});
        }
    std::sort(pairs.begin(), pairs.end(), [](const P &a, const P &b) { return a.s > b.s; });
    for (const P &p : pairs)
        if (match[p.d] < 0 && !used[p.t]) match[p.d] = int(p.t), used[p.t] = 1;
    pairs.clear();
    for (size_t i = 0; i < dets.size(); ++i) {
        if (match[i] >= 0) continue;
        const float cx = (dets[i].x0 + dets[i].x1) / 2, cy = (dets[i].y0 + dets[i].y1) / 2;
        const float sz = std::max(dets[i].x1 - dets[i].x0, dets[i].y1 - dets[i].y0);
        for (size_t j = 0; j < tracks_.size(); ++j) {
            if (used[j]) continue;
            const float *b = tracks_[j].box;
            const float d = std::hypot((b[0] + b[2]) / 2 - cx, (b[1] + b[3]) / 2 - cy) / std::max(sz, 1.f);
            if (d <= (weak(dets[i]) ? cfg_.dist_match_weak : cfg_.dist_match)) pairs.push_back({-d, i, j});
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const P &a, const P &b) { return a.s > b.s; });
    for (const P &p : pairs)
        if (match[p.d] < 0 && !used[p.t]) match[p.d] = int(p.t), used[p.t] = 1;
    std::vector<int> frame_state(tracks_.size(), kUnknown);
    for (size_t i = 0; i < dets.size(); ++i) {
        const LightDet &d = dets[i];
        if (match[i] >= 0) {
            Track &t = tracks_[size_t(match[i])];
            const float ocx = (t.box[0] + t.box[2]) / 2 - t.v[0], ocy = (t.box[1] + t.box[3]) / 2 - t.v[1];
            const float ncx = (d.x0 + d.x1) / 2, ncy = (d.y0 + d.y1) / 2;
            const float k = t.steps > 0 ? 1.f / t.steps : 1.f;  // frames since the last match
            t.v[0] = 0.5f * t.v[0] + 0.5f * (ncx - ocx) * k;
            t.v[1] = 0.5f * t.v[1] + 0.5f * (ncy - ocy) * k;
            t.box[0] = d.x0, t.box[1] = d.y0, t.box[2] = d.x1, t.box[3] = d.y1;
            t.score = d.score;
            t.horizontal = d.horizontal;
            ++t.hits;
            t.since = 0, t.steps = 0;
            observe(t, ts, d);
            frame_state[size_t(match[i])] = d.state;
        } else {
            if (weak(d)) continue;  // a weak box never starts a track
            Track t;
            t.id = next_id_++;
            t.box[0] = d.x0, t.box[1] = d.y0, t.box[2] = d.x1, t.box[3] = d.y1;
            t.score = d.score;
            t.hits = 1;
            t.horizontal = d.horizontal;
            t.state_t = ts;
            observe(t, ts, d);
            tracks_.push_back(t);
            frame_state.push_back(d.state);
        }
    }
    std::vector<Track> keep;
    std::vector<int> keep_fs;
    for (size_t j = 0; j < tracks_.size(); ++j) {
        // missed too often where it could be seen, or out of every view for long
        if (tracks_[j].since > cfg_.max_age || tracks_[j].steps > 3 * cfg_.max_age) continue;
        keep.push_back(std::move(tracks_[j]));
        keep_fs.push_back(frame_state[j]);
    }
    tracks_.swap(keep);
    for (size_t j = 0; j < tracks_.size(); ++j) {
        const Track &t = tracks_[j];
        // confirmed tracks: matched in this frame, or coasting for at most report_coast frames
        if (t.hits < cfg_.min_hits || t.steps > cfg_.report_coast) continue;
        LightTrackReport r;
        r.id = t.id;
        std::copy(t.box, t.box + 4, r.box);
        r.score = t.score;
        r.flashing = flashing(t, ts);
        r.state = r.flashing ? kYellow : t.state;
        r.state_conf = t.state_conf;
        r.frame_state = keep_fs[j];
        r.coast = t.steps;
        r.horizontal = t.horizontal;
        r.hits = t.hits;
        r.age = t.age;
        r.state_s = double(ts - t.state_t) * 1e-9;
        out.push_back(r);
    }
}

// ------------------------------------------------------------------ JSON ---

std::string road_users_json(uint64_t seq, int64_t ts_mono_ns, int64_t ts_real_ns, double ms, int frame_w, int frame_h,
                            const float *roi, const std::vector<RoadUserDet> &users) {
    char b[320];
    snprintf(b, sizeof(b),
             "{\"type\":\"road_users\",\"seq\":%" PRIu64 ",\"ts_mono_ns\":%" PRId64 ",\"ts_real_ns\":%" PRId64
             ",\"ms\":%s,\"frame\":[%d,%d],\"roi\":[%s,%s,%s,%s],\"view\":2",
             seq, ts_mono_ns, ts_real_ns, fmt(ms, 1).c_str(), frame_w, frame_h, fmt(roi[0] * frame_w, 0).c_str(),
             fmt(roi[1] * frame_h, 0).c_str(), fmt(roi[2] * frame_w, 0).c_str(), fmt(roi[3] * frame_h, 0).c_str());
    std::string s = b;
    // the same list as in a `lights` message: reuse its writer on an empty light message and cut the key out
    LightsConfig cfg;
    std::vector<LightTrackReport> none;
    const std::string l = lights_json(0, 0, 0, 0, frame_w, frame_h, cfg, none, nullptr, nullptr, 0, &users, 1);
    const size_t k = l.find(",\"road_users\":");
    s += l.substr(k, l.size() - 1 - k);
    s += "}";
    return s;
}

std::string lights_json(uint64_t seq, int64_t ts_mono_ns, int64_t ts_real_ns, double lights_ms, int frame_w,
                        int frame_h, const LightsConfig &cfg, const std::vector<LightTrackReport> &tracks,
                        const std::vector<LightDet> *raw, const LightsTiming *tm, int view,
                        const std::vector<RoadUserDet> *users, int users_state) {
    const bool v2 = cfg.roi2[2] > cfg.roi2[0] && cfg.roi2[3] > cfg.roi2[1];
    const float *roi = view == 1 && v2 ? cfg.roi2 : cfg.roi;
    char b[512];
    snprintf(b, sizeof(b),
             "{\"type\":\"lights\",\"seq\":%" PRIu64 ",\"ts_mono_ns\":%" PRId64 ",\"ts_real_ns\":%" PRId64
             ",\"lights_ms\":%s,\"frame\":[%d,%d],\"roi\":[%s,%s,%s,%s],\"view\":%d,\"lights\":[",
             seq, ts_mono_ns, ts_real_ns, fmt(lights_ms, 1).c_str(), frame_w, frame_h, fmt(roi[0] * frame_w, 0).c_str(),
             fmt(roi[1] * frame_h, 0).c_str(), fmt(roi[2] * frame_w, 0).c_str(), fmt(roi[3] * frame_h, 0).c_str(),
             view == 1 && v2 ? 1 : 0);
    std::string s = b;
    for (size_t i = 0; i < tracks.size(); ++i) {
        const LightTrackReport &t = tracks[i];
        snprintf(b, sizeof(b),
                 "%s{\"track_id\":%d,\"state\":\"%s\",\"state_conf\":%s,\"state_s\":%s,\"frame_state\":\"%s\",\"flashing\":%s,"
                 "\"score\":%s,\"box\":[%s,%s,%s,%s],\"horizontal\":%s,\"hits\":%d,\"age\":%d,\"coast\":%d}",
                 i ? "," : "", t.id, light_state_name(t.state), fmt(t.state_conf, 2).c_str(), fmt(t.state_s, 1).c_str(),
                 light_state_name(t.frame_state), t.flashing ? "true" : "false", fmt(t.score, 3).c_str(),
                 fmt(t.box[0], 1).c_str(), fmt(t.box[1], 1).c_str(), fmt(t.box[2], 1).c_str(), fmt(t.box[3], 1).c_str(),
                 t.horizontal ? "true" : "false", t.hits, t.age, t.coast);
        s += b;
    }
    s += "]";
    if (raw) {
        s += ",\"raw\":[";
        for (size_t i = 0; i < raw->size(); ++i) {
            const LightDet &d = (*raw)[i];
            snprintf(b, sizeof(b), "%s{\"box\":[%s,%s,%s,%s],\"score\":%s,\"state\":\"%s\",\"state_conf\":%s,\"rule\":[%s,%s,%s]}",
                     i ? "," : "", fmt(d.x0, 1).c_str(), fmt(d.y0, 1).c_str(), fmt(d.x1, 1).c_str(), fmt(d.y1, 1).c_str(),
                     fmt(d.score, 3).c_str(), light_state_name(d.state), fmt(d.state_conf, 2).c_str(),
                     fmt(d.rule[0], 4).c_str(), fmt(d.rule[1], 4).c_str(), fmt(d.rule[2], 4).c_str());
            s += b;
        }
        s += "]";
    }
    if (users_state == 2) {
        s += ",\"road_users\":null";
    } else if (users_state == 1 && users) {
        // {"cls", "score", "box" [x0,y0,x1,y1] frame px, "clip" (edges of the view's region the box touches: l t r b),
        //  "rider" (bike / motor with its rider paired), "p" (the eight class scores, BDD100K order)}
        s += ",\"road_users\":[";
        for (size_t i = 0; i < users->size(); ++i) {
            const RoadUserDet &d = (*users)[i];
            char clip[5];
            int n = 0;
            if (d.clip & 1) clip[n++] = 'l';
            if (d.clip & 2) clip[n++] = 't';
            if (d.clip & 4) clip[n++] = 'r';
            if (d.clip & 8) clip[n++] = 'b';
            clip[n] = 0;
            snprintf(b, sizeof(b),
                     "%s{\"cls\":\"%s\",\"score\":%s,\"box\":[%s,%s,%s,%s],\"clip\":\"%s\",\"rider\":%s,"
                     "\"p\":[%s,%s,%s,%s,%s,%s,%s,%s]}",
                     i ? "," : "", road_user_name(d.cls), fmt(d.score, 3).c_str(), fmt(d.x0, 1).c_str(), fmt(d.y0, 1).c_str(),
                     fmt(d.x1, 1).c_str(), fmt(d.y1, 1).c_str(), clip, d.rider ? "true" : "false", fmt(d.p[0], 2).c_str(),
                     fmt(d.p[1], 2).c_str(), fmt(d.p[2], 2).c_str(), fmt(d.p[3], 2).c_str(), fmt(d.p[4], 2).c_str(),
                     fmt(d.p[5], 2).c_str(), fmt(d.p[6], 2).c_str(), fmt(d.p[7], 2).c_str());
            s += b;
        }
        s += "]";
    }
    if (tm) {
        snprintf(b, sizeof(b), ",\"timing\":{\"pre_ms\":%s,\"det_ms\":%s,\"post_ms\":%s,\"state_ms\":%s,\"wait_ms\":%s,\"users_ms\":%s}",
                 fmt(tm->pre_ms, 2).c_str(), fmt(tm->det_ms, 2).c_str(), fmt(tm->post_ms, 2).c_str(),
                 fmt(tm->state_ms, 2).c_str(), fmt(tm->wait_ms, 2).c_str(), fmt(tm->users_ms, 3).c_str());
        s += b;
    }
    s += "}";
    return s;
}

}  // namespace signs
}  // namespace roadsense
