#include "tracker.h"

#include "verify.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace roadsense {
namespace signs {

namespace {

void to_z(const float b[4], double z[4]) {
    const double w = std::max(1.0f, b[2] - b[0]), h = std::max(1.0f, b[3] - b[1]);
    z[0] = b[0] + w / 2;
    z[1] = b[1] + h / 2;
    z[2] = w * h;
    z[3] = w / h;
}

float iou4(const float a[4], const float b[4]) {
    const float ix = std::max(0.f, std::min(a[2], b[2]) - std::max(a[0], b[0]));
    const float iy = std::max(0.f, std::min(a[3], b[3]) - std::max(a[1], b[1]));
    const float inter = ix * iy;
    const float u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return u > 0 ? inter / u : 0.f;
}

// In-place inverse of a 4x4 SPD matrix (Gauss-Jordan with partial pivoting).
bool inv4(double a[4][4]) {
    double m[4][8];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 8; ++j) m[i][j] = j < 4 ? a[i][j] : (j - 4 == i ? 1.0 : 0.0);
    for (int c = 0; c < 4; ++c) {
        int p = c;
        for (int r = c + 1; r < 4; ++r)
            if (std::fabs(m[r][c]) > std::fabs(m[p][c])) p = r;
        if (std::fabs(m[p][c]) < 1e-12) return false;
        if (p != c)
            for (int j = 0; j < 8; ++j) std::swap(m[p][j], m[c][j]);
        const double d = 1.0 / m[c][c];
        for (int j = 0; j < 8; ++j) m[c][j] *= d;
        for (int r = 0; r < 4; ++r) {
            if (r == c) continue;
            const double f = m[r][c];
            if (f != 0)
                for (int j = 0; j < 8; ++j) m[r][j] -= f * m[c][j];
        }
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) a[i][j] = m[i][j + 4];
    return true;
}

}  // namespace

// ------------------------------------------------------------------ Kalman ---
// x = [cx, cy, s(area), r(aspect w/h), vcx, vcy, vs], z = [cx, cy, s, r].
// Noise is relative to the box size (SORT uses absolute px values tuned for
// 30 fps pedestrians; signs span 10 to 300 px here).

void Tracker::Kalman::init(const float b[4]) {
    double z[4];
    to_z(b, z);
    std::memset(x, 0, sizeof(x));
    std::memset(P, 0, sizeof(P));
    for (int i = 0; i < 4; ++i) x[i] = z[i];
    const double L = std::sqrt(z[2]);  // box size scale (px)
    P[0][0] = P[1][1] = (0.1 * L) * (0.1 * L);
    P[2][2] = (0.1 * z[2]) * (0.1 * z[2]);
    P[3][3] = 0.01;
    P[4][4] = P[5][5] = (1.0 * L) * (1.0 * L);  // unknown velocity: up to ~a box per step
    P[6][6] = (0.5 * z[2]) * (0.5 * z[2]);
}

void Tracker::Kalman::predict() {
    if (x[2] + x[6] <= 0) x[6] = 0;
    x[0] += x[4];
    x[1] += x[5];
    x[2] += x[6];
    // P = F P F^T with F = I + E(0,4) + E(1,5) + E(2,6)
    double T[7][7];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j) T[i][j] = P[i][j] + (i < 3 ? P[i + 4][j] : 0.0);
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j) P[i][j] = T[i][j] + (j < 3 ? T[i][j + 4] : 0.0);
    const double s = std::max(x[2], 1.0), L = std::sqrt(s);
    const double q[7] = {(0.05 * L) * (0.05 * L), (0.05 * L) * (0.05 * L), (0.05 * s) * (0.05 * s), 1e-3,
                         (0.3 * L) * (0.3 * L),   (0.3 * L) * (0.3 * L),   (0.2 * s) * (0.2 * s)};
    for (int i = 0; i < 7; ++i) P[i][i] += q[i];
}

void Tracker::Kalman::update(const float b[4]) {
    double z[4];
    to_z(b, z);
    const double L = std::sqrt(z[2]);
    const double R[4] = {(0.05 * L) * (0.05 * L) + 1.0, (0.05 * L) * (0.05 * L) + 1.0, (0.1 * z[2]) * (0.1 * z[2]),
                         0.01};
    double S[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) S[i][j] = P[i][j] + (i == j ? R[i] : 0.0);
    if (!inv4(S)) return;
    double K[7][4];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 4; ++j) {
            double a = 0;
            for (int k = 0; k < 4; ++k) a += P[i][k] * S[k][j];
            K[i][j] = a;
        }
    double y[4];
    for (int i = 0; i < 4; ++i) y[i] = z[i] - x[i];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 4; ++j) x[i] += K[i][j] * y[j];
    // P = (I - K H) P, H selects the first 4 states
    double N[7][7];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j) {
            double a = P[i][j];
            for (int k = 0; k < 4; ++k) a -= K[i][k] * P[k][j];
            N[i][j] = a;
        }
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j) P[i][j] = N[i][j];
}

void Tracker::Kalman::box(float out[4]) const {
    const double s = std::max(x[2], 1.0), r = std::max(x[3], 1e-3);
    const double w = std::sqrt(s * r), h = s / w;
    out[0] = float(x[0] - w / 2);
    out[1] = float(x[1] - h / 2);
    out[2] = float(x[0] + w / 2);
    out[3] = float(x[1] + h / 2);
}

// ----------------------------------------------------------------- Tracker ---

bool parse_class_min_score(const std::string &s, std::vector<std::pair<std::string, float>> *out) {
    out->clear();
    size_t i = 0;
    while (i < s.size()) {
        size_t e = s.find(',', i);
        if (e == std::string::npos) e = s.size();
        const std::string item = s.substr(i, e - i);
        const size_t eq = item.find('=');
        if (eq == std::string::npos || eq == 0) return false;
        char *end = nullptr;
        const float v = strtof(item.c_str() + eq + 1, &end);
        if (end == item.c_str() + eq + 1) return false;
        out->emplace_back(item.substr(0, eq), v);
        i = e + 1;
    }
    return true;
}

Tracker::Tracker(const TrackerConfig &cfg, const ClassNames &names) : cfg_(cfg), names_(names) {
    red_merged_.assign(names_.merged.size(), 0);
    min_score_merged_.assign(names_.merged.size(), 0.f);
    for (size_t k = 0; k < names_.merged.size(); ++k) {
        red_merged_[k] = is_round_red(names_.merged[k]);
        size_t best_len = 0;
        for (const auto &p : cfg_.class_min_score)
            if (names_.merged[k].compare(0, p.first.size(), p.first) == 0 && p.first.size() >= best_len)
                best_len = p.first.size(), min_score_merged_[k] = p.second;
    }
}

int Tracker::merged_index(const Track &t) const {
    if (t.nvotes == 0) return -1;
    return int(std::max_element(t.votes.begin(), t.votes.end()) - t.votes.begin());
}

const char *Tracker::suppress_reason(const Track &t, int k) const {
    if (k < 0) return nullptr;
    // Classifiers trained with a "not a sign" class (roadsense L12, docs/SIGN_MODELS.md section 7) vote it like
    // any other class; a track whose vote is `background` is reported flagged, never as a sign.
    if (names_.merged[size_t(k)] == "background") return "not_a_sign";
    if (red_merged_[size_t(k)]) {
        const bool verified = t.n_verified >= cfg_.verify_min;
        if (verified && t.verify_sum < cfg_.verify_thr * t.n_verified) return "verifier";
        const bool glowing = t.n_glow_checked >= cfg_.glow_min && t.n_glow >= cfg_.glow_share * t.n_glow_checked;
        if (glowing && (verified ? t.verify_sum < cfg_.verify_vouch * t.n_verified : cfg_.glow_alone)) return "glow";
    }
    if (t.best_score < min_score_merged_[size_t(k)]) return "score";
    return nullptr;
}

void Tracker::classify(const Track &t, std::string *cls, int *cls_id) const {
    if (t.nvotes == 0) {  // detector saw a sign, classifier never confident
        *cls = "traffic-sign";
        *cls_id = -1;
        return;
    }
    const int k = int(std::max_element(t.votes.begin(), t.votes.end()) - t.votes.begin());
    *cls = names_.merged[size_t(k)];
    *cls_id = names_.merged_to_raw[size_t(k)];
}

void Tracker::end(Track &t, std::vector<EndedTrack> &ended) {
    if (!t.confirmed) return;
    EndedTrack e;
    e.id = t.id;
    classify(t, &e.cls, &e.cls_id);
    e.first_seq = t.first_seq;
    e.last_seq = t.last_seq;
    e.hits = t.hits;
    e.best_score = t.best_score;
    e.reason = suppress_reason(t, merged_index(t));
    e.suppressed = e.reason != nullptr;
    e.glow_share = t.n_glow_checked ? float(t.n_glow) / float(t.n_glow_checked) : -1.f;
    e.verify = t.n_verified ? t.verify_sum / float(t.n_verified) : -1.f;
    e.crop = std::move(t.crop);
    ended.push_back(std::move(e));
}

void Tracker::update(uint64_t seq, const Nv12View &frame, const std::vector<Detection> &dets,
                     std::vector<TrackReport> &reports, std::vector<EndedTrack> &ended) {
    reports.clear();
    for (Track &t : tracks_) {
        t.kf.predict();
        t.kf.box(t.pred);
        ++t.age;
        ++t.since_update;
    }
    const size_t nt = tracks_.size(), nd = dets.size();
    std::vector<int> det_track(nd, -1);
    std::vector<char> tused(nt, 0);

    struct Pair {
        float cost;
        int t, d;
    };
    std::vector<Pair> pairs;
    // pass 1: IoU
    for (size_t i = 0; i < nt; ++i)
        for (size_t j = 0; j < nd; ++j) {
            const float b[4] = {dets[j].x0, dets[j].y0, dets[j].x1, dets[j].y1};
            const float v = iou4(tracks_[i].pred, b);
            if (v >= cfg_.iou_match) pairs.push_back({-v, int(i), int(j)});
        }
    auto assign = [&]() {
        std::sort(pairs.begin(), pairs.end(), [](const Pair &a, const Pair &b) { return a.cost < b.cost; });
        for (const Pair &p : pairs)
            if (!tused[size_t(p.t)] && det_track[size_t(p.d)] < 0) {
                tused[size_t(p.t)] = 1;
                det_track[size_t(p.d)] = p.t;
            }
        pairs.clear();
    };
    assign();
    // pass 2: centre distance for the leftovers, with a size-ratio sanity check
    for (size_t i = 0; i < nt; ++i) {
        if (tused[i]) continue;
        const float *p = tracks_[i].pred;
        const float pw = p[2] - p[0], ph = p[3] - p[1];
        for (size_t j = 0; j < nd; ++j) {
            if (det_track[j] >= 0) continue;
            const Detection &d = dets[j];
            const float dw = d.x1 - d.x0, dh = d.y1 - d.y0;
            const float ratio = std::sqrt((dw * dh) / std::max(1.f, pw * ph));
            if (ratio < 0.5f || ratio > 2.0f) continue;
            const float dx = (d.x0 + d.x1 - p[0] - p[2]) / 2, dy = (d.y0 + d.y1 - p[1] - p[3]) / 2;
            const float dist = std::sqrt(dx * dx + dy * dy) / std::max({pw, ph, dw, dh, 1.f});
            if (dist <= cfg_.dist_match) pairs.push_back({dist, int(i), int(j)});
        }
    }
    assign();

    const size_t n_merged = names_.merged.size();
    auto absorb = [&](Track &t, const Detection &d) {
        const float b[4] = {d.x0, d.y0, d.x1, d.y1};
        ++t.hits;
        t.since_update = 0;
        t.last_seq = seq;
        if (t.hits >= cfg_.min_hits) t.confirmed = true;
        if (d.cls_raw >= 0 && d.cls_prob >= cfg_.cls_gate && d.probs.size() == names_.raw.size()) {
            for (size_t k = 0; k < d.probs.size(); ++k) t.votes[size_t(names_.raw_to_merged[k])] += d.probs[k];
            ++t.nvotes;
        }
        if (d.glow >= 0) {
            ++t.n_glow_checked;
            t.n_glow += d.glow >= cfg_.glow_px;
        }
        if (d.verify >= 0) {
            ++t.n_verified;
            t.verify_sum += d.verify;
        }
        if (d.score > t.best_score) {
            t.best_score = d.score;
            const float mw = cfg_.crop_margin * (b[2] - b[0]), mh = cfg_.crop_margin * (b[3] - b[1]);
            nv12_crop(frame, int(std::floor(b[0] - mw)), int(std::floor(b[1] - mh)), int(std::ceil(b[2] + mw)),
                      int(std::ceil(b[3] + mh)), t.crop);
        }
    };
    for (size_t j = 0; j < nd; ++j) {
        const Detection &d = dets[j];
        const float b[4] = {d.x0, d.y0, d.x1, d.y1};
        if (det_track[j] >= 0) {
            Track &t = tracks_[size_t(det_track[j])];
            t.kf.update(b);
            absorb(t, d);
        } else {
            Track t;
            t.id = next_id_++;
            t.kf.init(b);
            t.first_seq = seq;
            t.votes.assign(n_merged, 0.f);
            absorb(t, d);
            tracks_.push_back(std::move(t));
            det_track[j] = int(tracks_.size()) - 1;
        }
    }
    // Reports: confirmed tracks matched in this frame, in detection order.
    for (size_t j = 0; j < nd; ++j) {
        const Track &t = tracks_[size_t(det_track[j])];
        if (!t.confirmed) continue;
        TrackReport r;
        r.id = t.id;
        classify(t, &r.cls, &r.cls_id);
        r.score = dets[j].score;
        r.box[0] = dets[j].x0;
        r.box[1] = dets[j].y0;
        r.box[2] = dets[j].x1;
        r.box[3] = dets[j].y1;
        r.age = t.age;
        r.hits = t.hits;
        r.reason = suppress_reason(t, merged_index(t));
        r.suppressed = r.reason != nullptr;
        r.glow_share = t.n_glow_checked ? float(t.n_glow) / float(t.n_glow_checked) : -1.f;
        r.verify = t.n_verified ? t.verify_sum / float(t.n_verified) : -1.f;
        reports.push_back(std::move(r));
    }
    // Reap.
    for (size_t i = 0; i < tracks_.size();) {
        if (tracks_[i].since_update > cfg_.max_age) {
            end(tracks_[i], ended);
            tracks_.erase(tracks_.begin() + long(i));
        } else {
            ++i;
        }
    }
}

void Tracker::flush(std::vector<EndedTrack> &ended) {
    for (Track &t : tracks_) end(t, ended);
    tracks_.clear();
}

}  // namespace signs
}  // namespace roadsense
