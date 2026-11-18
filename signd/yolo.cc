#include "yolo.h"

#include "verify.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>

namespace roadsense {
namespace signs {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ------------------------------------------------------------ resampling ---
// Separable triangle-filter resampling with PIL's BILINEAR semantics: for a
// downscale by s the filter support is widened to s (antialiasing), for an
// upscale it is plain bilinear. Weights are Q14 fixed point. Taps outside the
// plane are dropped and the rest renormalised (PIL's edge behaviour).

constexpr int kWBits = 14;
constexpr int kWOne = 1 << kWBits;

struct Taps {
    std::vector<int> start, count;
    std::vector<int16_t> w;  // count[i] weights at i*maxn, Q14
    std::vector<uint8_t> w7; // the same in Q7 (sum exactly 128) for the uint16 vertical pass
    int maxn = 0;
};

void compute_taps(double s0, double len, int src_len, int dst_len, Taps &t) {
    const double scale = len / dst_len;
    const double fs = std::max(scale, 1.0);
    const double support = fs;  // triangle radius 1, scaled
    t.maxn = int(std::ceil(support)) * 2 + 2;
    t.start.assign(dst_len, 0);
    t.count.assign(dst_len, 0);
    t.w.assign(size_t(dst_len) * t.maxn, 0);
    t.w7.assign(size_t(dst_len) * t.maxn, 0);
    std::vector<double> ww(t.maxn);
    for (int i = 0; i < dst_len; ++i) {
        const double center = s0 + (i + 0.5) * scale;
        int xmin = int(std::floor(center - support + 0.5));
        int xmax = int(std::floor(center + support + 0.5));
        xmin = std::max(xmin, 0);
        xmax = std::min(xmax, src_len);
        if (xmax - xmin > t.maxn) xmax = xmin + t.maxn;
        if (xmax <= xmin) {  // entirely outside: nearest edge pixel
            xmin = std::min(std::max(int(std::floor(center)), 0), src_len - 1);
            xmax = xmin + 1;
        }
        double sum = 0;
        int n = xmax - xmin;
        for (int k = 0; k < n; ++k) {
            const double d = std::fabs((xmin + k + 0.5 - center) / fs);
            ww[k] = d < 1.0 ? 1.0 - d : 0.0;
            sum += ww[k];
        }
        if (sum <= 0) {
            for (int k = 0; k < n; ++k) ww[k] = 1.0;
            sum = n;
        }
        int isum = 0, kmax = 0;
        for (int k = 0; k < n; ++k) {
            const int q = int(std::lround(ww[k] / sum * kWOne));
            t.w[size_t(i) * t.maxn + k] = int16_t(q);
            isum += q;
            if (ww[k] > ww[kmax]) kmax = k;
        }
        t.w[size_t(i) * t.maxn + kmax] = int16_t(t.w[size_t(i) * t.maxn + kmax] + (kWOne - isum));
        int s7 = 0;
        for (int k = 0; k < n; ++k) {
            const int q = (t.w[size_t(i) * t.maxn + k] + 64) >> 7;
            t.w7[size_t(i) * t.maxn + k] = uint8_t(q);
            s7 += q;
        }
        t.w7[size_t(i) * t.maxn + kmax] = uint8_t(t.w7[size_t(i) * t.maxn + kmax] + (128 - s7));
        t.start[i] = xmin;
        t.count[i] = n;
    }
}

inline uint8_t clamp_u8(int v) { return uint8_t(v < 0 ? 0 : (v > 255 ? 255 : v)); }

// Resamples a plane of nch interleaved channels (1 = Y, 2 = NV12 CbCr) to
// dw x dh, one uint8 output plane per channel. Vertical pass first: it runs
// over contiguous bytes (all channels at once, vectorised) and leaves only dh
// rows for the scalar horizontal pass, which is the cheap order for the large
// downscale of the detector letterbox. Only the columns the horizontal taps
// touch are processed, so classifier crops cost only their own area.
void resample_plane(const uint8_t *src, int stride, int nch, int src_w, int src_h, double sx, double sy, double sw,
                    double sh, int dw, int dh, uint8_t *const *dst) {
    thread_local Taps th, tv;
    thread_local std::vector<uint8_t> tmp;
    thread_local std::vector<uint16_t> acc;
    compute_taps(sx, sw, src_w, dw, th);
    compute_taps(sy, sh, src_h, dh, tv);
    const int c0 = th.start[0];
    int c1 = c0;
    for (int i = 0; i < dw; ++i) c1 = std::max(c1, th.start[i] + th.count[i]);
    const int rowb = (c1 - c0) * nch;  // bytes per intermediate row
    if (tmp.size() < size_t(rowb) * dh) tmp.resize(size_t(rowb) * dh);
    if (acc.size() < size_t(rowb)) acc.resize(size_t(rowb));
    // Vertical: Q7 weights summing to 128, so 128*255 + 64 fits uint16 and
    // the loop maps onto 8/16-lane widening multiply-accumulate (NEON umlal).
    uint16_t *__restrict A = acc.data();
    for (int j = 0; j < dh; ++j) {
        for (int i = 0; i < rowb; ++i) A[i] = 64;
        const uint8_t *w = &tv.w7[size_t(j) * tv.maxn];
        for (int k = 0; k < tv.count[j]; ++k) {
            const uint8_t *__restrict row = src + size_t(tv.start[j] + k) * stride + size_t(c0) * nch;
            const uint16_t wk = w[k];
            for (int i = 0; i < rowb; ++i) A[i] = uint16_t(A[i] + wk * row[i]);
        }
        uint8_t *__restrict o = tmp.data() + size_t(j) * rowb;
        for (int i = 0; i < rowb; ++i) o[i] = uint8_t(A[i] >> 7);
    }
    for (int ch = 0; ch < nch; ++ch) {
        for (int j = 0; j < dh; ++j) {
            const uint8_t *row = tmp.data() + size_t(j) * rowb + ch;
            uint8_t *o = dst[ch] + size_t(j) * dw;
            for (int i = 0; i < dw; ++i) {
                const int16_t *w = &th.w[size_t(i) * th.maxn];
                const uint8_t *p = row + size_t(th.start[i] - c0) * nch;
                int a = 1 << (kWBits - 1);
                const int n = th.count[i];
                for (int k = 0; k < n; ++k) a += w[k] * p[k * nch];
                o[i] = clamp_u8(a >> kWBits);
            }
        }
    }
}

float iou(const Detection &a, const Detection &b) {
    const float ix = std::max(0.f, std::min(a.x1, b.x1) - std::max(a.x0, b.x0));
    const float iy = std::max(0.f, std::min(a.y1, b.y1) - std::max(a.y0, b.y0));
    const float inter = ix * iy;
    const float ua = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter;
    return inter / (ua + 1e-9f);
}

}  // namespace

// ----------------------------------------------------------- image helpers ---

void nv12_region_to_rgb_planar(const Nv12View &src, double sx, double sy, double sw, double sh, int dw, int dh,
                               float *dst, int dst_w, size_t dst_plane, int dx, int dy) {
    const size_t n = size_t(dw) * dh;
    thread_local std::vector<uint8_t> planes;
    if (planes.size() < 3 * n) planes.resize(3 * n);
    uint8_t *Y = planes.data(), *U = Y + n, *V = U + n;
    resample_plane(src.y, src.y_stride, 1, src.width, src.height, sx, sy, sw, sh, dw, dh, &Y);
    // Chroma sample c covers luma [2c, 2c+2): the same rectangle in chroma pixels is halved.
    uint8_t *const uvp[2] = {U, V};
    resample_plane(src.uv, src.uv_stride, 2, src.width / 2, src.height / 2, sx / 2, sy / 2, sw / 2, sh / 2, dw, dh,
                   uvp);
    float *R = dst, *G = dst + dst_plane, *B = dst + 2 * dst_plane;
    constexpr float k = 1.0f / 255.0f;
    for (int j = 0; j < dh; ++j) {
        const size_t so = size_t(j) * dw, oo = size_t(dy + j) * dst_w + dx;
        for (int i = 0; i < dw; ++i) {
            const float y = Y[so + i], u = float(U[so + i]) - 128.f, v = float(V[so + i]) - 128.f;
            const float r = y + 1.402f * v, g = y - 0.344136f * u - 0.714136f * v, b = y + 1.772f * u;
            R[oo + i] = std::min(std::max(r, 0.f), 255.f) * k;
            G[oo + i] = std::min(std::max(g, 0.f), 255.f) * k;
            B[oo + i] = std::min(std::max(b, 0.f), 255.f) * k;
        }
    }
}

void nv12_crop(const Nv12View &src, int x0, int y0, int x1, int y1, Nv12Image &out) {
    x0 = std::max(0, x0) & ~1;
    y0 = std::max(0, y0) & ~1;
    x1 = std::min(src.width, (x1 + 1) & ~1);
    y1 = std::min(src.height, (y1 + 1) & ~1);
    if (x1 - x0 < 2) x1 = std::min(src.width, x0 + 2);
    if (y1 - y0 < 2) y1 = std::min(src.height, y0 + 2);
    out.width = x1 - x0;
    out.height = y1 - y0;
    out.data.resize(size_t(out.width) * out.height * 3 / 2);
    uint8_t *y = out.data.data(), *uv = y + size_t(out.width) * out.height;
    for (int r = y0; r < y1; ++r)
        std::copy_n(src.y + size_t(r) * src.y_stride + x0, out.width, y + size_t(r - y0) * out.width);
    for (int r = y0 / 2; r < y1 / 2; ++r)
        std::copy_n(src.uv + size_t(r) * src.uv_stride + x0, out.width, uv + size_t(r - y0 / 2) * out.width);
}

void rgb_to_nv12(const uint8_t *rgb, int width, int height, Nv12Image &out) {
    const int w = width & ~1, h = height & ~1;
    out.width = w;
    out.height = h;
    out.data.resize(size_t(w) * h * 3 / 2);
    uint8_t *Y = out.data.data(), *UV = Y + size_t(w) * h;
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c) {
            const uint8_t *p = rgb + (size_t(r) * width + c) * 3;
            Y[size_t(r) * w + c] = clamp_u8(int(std::lround(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2])));
        }
    for (int r = 0; r < h / 2; ++r)
        for (int c = 0; c < w / 2; ++c) {
            double R = 0, G = 0, B = 0;
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                    const uint8_t *p = rgb + (size_t(2 * r + dy) * width + 2 * c + dx) * 3;
                    R += p[0];
                    G += p[1];
                    B += p[2];
                }
            R /= 4;
            G /= 4;
            B /= 4;
            UV[size_t(r) * w + 2 * c] = clamp_u8(int(std::lround(-0.168736 * R - 0.331264 * G + 0.5 * B + 128)));
            UV[size_t(r) * w + 2 * c + 1] = clamp_u8(int(std::lround(0.5 * R - 0.418688 * G - 0.081312 * B + 128)));
        }
}

void nv12_to_rgb(const Nv12View &src, std::vector<uint8_t> &rgb) {
    rgb.resize(size_t(src.width) * src.height * 3);
    for (int r = 0; r < src.height; ++r)
        for (int c = 0; c < src.width; ++c) {
            const float y = src.y[size_t(r) * src.y_stride + c];
            const uint8_t *uv = src.uv + size_t(r / 2) * src.uv_stride + (c & ~1);
            const float u = float(uv[0]) - 128.f, v = float(uv[1]) - 128.f;
            uint8_t *o = &rgb[(size_t(r) * src.width + c) * 3];
            o[0] = clamp_u8(int(std::lround(y + 1.402f * v)));
            o[1] = clamp_u8(int(std::lround(y - 0.344136f * u - 0.714136f * v)));
            o[2] = clamp_u8(int(std::lround(y + 1.772f * u)));
        }
}

std::string strip_variant(const std::string &name) {
    // Remove a trailing "--g<digits>".
    const size_t p = name.rfind("--g");
    if (p == std::string::npos || p + 3 >= name.size()) return name;
    for (size_t i = p + 3; i < name.size(); ++i)
        if (name[i] < '0' || name[i] > '9') return name;
    return name.substr(0, p);
}

bool ClassNames::load(const std::string &path, std::string *err) {
    std::ifstream f(path);
    if (!f) {
        if (err) *err = "cannot read " + path;
        return false;
    }
    raw.clear();
    merged.clear();
    raw_to_merged.clear();
    merged_to_raw.clear();
    std::map<std::string, int> idx;
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        const int id = int(raw.size());
        raw.push_back(line);
        const std::string m = strip_variant(line);
        auto it = idx.find(m);
        if (it == idx.end()) {
            it = idx.emplace(m, int(merged.size())).first;
            merged.push_back(m);
            merged_to_raw.push_back(id);
        }
        raw_to_merged.push_back(it->second);
    }
    if (raw.empty()) {
        if (err) *err = "empty class list " + path;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------- pipeline ---

std::string default_model_path(const std::string &models_dir, const std::string &which, const std::string &backend) {
    const std::string stem = which == "det" ? "mtsd_signdet_yolov8n_384x640" : "mtsd_signcls_yolov8n_64";
    return models_dir + "/" + stem + (backend == "ncnn" ? "-ncnn.param" : ".onnx");
}

SignPipeline::SignPipeline(const PipelineConfig &cfg) : cfg_(cfg) {
    if (cfg_.det_model.empty()) cfg_.det_model = default_model_path(cfg_.models_dir, "det", cfg_.det_backend);
    if (cfg_.cls_model.empty()) cfg_.cls_model = default_model_path(cfg_.models_dir, "cls", cfg_.cls_backend);
    if (cfg_.cls_names.empty()) cfg_.cls_names = cfg_.models_dir + "/mtsd_signcls_classes.txt";
    std::string err;
    if (!names_.load(cfg_.cls_names, &err)) throw std::runtime_error(err);
    in_w_ = cfg_.det_w;
    in_h_ = cfg_.det_h;
    cls_s_ = cfg_.cls_size;
    n_cls_ = int(names_.raw.size());
    n_out_ = 4 + 1;  // cx, cy, w, h, one class score (single-class detector)
    n_anchor_ = 0;
    for (int s : {8, 16, 32}) n_anchor_ += (in_w_ / s) * (in_h_ / s);

    RunnerOptions ro;
    ro.threads = cfg_.threads;
    ro.allow_spinning = cfg_.allow_spinning;
    ro.ncnn_fp16 = cfg_.ncnn_fp16;
    ro.ncnn_flushall = cfg_.ncnn_flushall;
    ro.backend = cfg_.det_backend;
    ro.model = cfg_.det_model;
    ro.input_shape = {1, 3, in_h_, in_w_};
    ro.output_size = size_t(n_out_) * n_anchor_;
    det_ = make_runner(ro);
    ro.backend = cfg_.cls_backend;
    ro.model = cfg_.cls_model;
    ro.input_shape = {1, 3, cls_s_, cls_s_};
    ro.output_size = size_t(n_cls_);
    cls_ = make_runner(ro);
    red_raw_.assign(names_.raw.size(), 0);
    for (size_t i = 0; i < names_.raw.size(); ++i) red_raw_[i] = is_round_red(strip_variant(names_.raw[i]));
    app_in_.assign(size_t(3) * kAppearanceSize * kAppearanceSize, 0.f);
    if (!cfg_.verifier_model.empty() && !std::ifstream(cfg_.verifier_model).good()) {
        // The verifier's weights are never shipped (non-commercial training data): a missing file is not fatal.
        fprintf(stderr, "signd: verifier %s not found, running without it\n", cfg_.verifier_model.c_str());
        cfg_.verifier_model.clear();
    }
    if (!cfg_.verifier_model.empty()) {
        ro.backend = "ort";
        ro.model = cfg_.verifier_model;
        ro.input_shape = {1, 3, kAppearanceSize, kAppearanceSize};
        ro.output_size = 1;
        ver_ = make_runner(ro);
    }

    det_in_.assign(size_t(3) * in_w_ * in_h_, 114.f / 255.f);
    det_out_.assign(size_t(n_out_) * n_anchor_, 0.f);
    cls_in_.assign(size_t(3) * cls_s_ * cls_s_, 0.f);
    cls_out_.assign(size_t(n_cls_), 0.f);
}

SignPipeline::~SignPipeline() = default;

void SignPipeline::classify_locked(const Nv12View &f, std::vector<Detection> &out, double *app_ms) {
    const size_t cplane = size_t(cls_s_) * cls_s_;
    for (Detection &d : out) {
        const double bw = d.x1 - d.x0, bh = d.y1 - d.y0, m = cfg_.cls_margin;
        double x0 = std::max(0.0, d.x0 - m * bw), y0 = std::max(0.0, d.y0 - m * bh);
        double x1 = std::min(double(f.width), d.x1 + m * bw), y1 = std::min(double(f.height), d.y1 + m * bh);
        const double w = x1 - x0, h = y1 - y0;
        // Ultralytics classify transform: short side -> S, centre crop S x S.
        const double s = cls_s_ / std::min(w, h);
        const int rw = std::max(cls_s_, int(std::lround(w * s))), rh = std::max(cls_s_, int(std::lround(h * s)));
        const int l = (rw - cls_s_) / 2, tp = (rh - cls_s_) / 2;
        const double sxs = w / rw, sys = h / rh;  // source px per resized px
        nv12_region_to_rgb_planar(f, x0 + l * sxs, y0 + tp * sys, cls_s_ * sxs, cls_s_ * sys, cls_s_, cls_s_,
                                  cls_in_.data(), cls_s_, cplane, 0, 0);
        cls_->run(cls_in_.data(), cls_out_.data());
        const int k = int(std::max_element(cls_out_.begin(), cls_out_.end()) - cls_out_.begin());
        d.cls_raw = k;
        d.cls_prob = cls_out_[size_t(k)];
        d.probs = cls_out_;
        if ((cfg_.appearance || ver_) && red_raw_[size_t(k)]) {
            const auto ta = Clock::now();
            nv12_region_to_rgb_planar(f, d.x0, d.y0, d.x1 - d.x0, d.y1 - d.y0, kAppearanceSize, kAppearanceSize,
                                      app_in_.data(), kAppearanceSize, size_t(kAppearanceSize) * kAppearanceSize, 0, 0);
            if (cfg_.appearance) d.glow = glow_score(app_in_.data(), kAppearanceSize);
            if (ver_) {
                float logit = 0.f;
                ver_->run(app_in_.data(), &logit);
                d.verify = 1.f / (1.f + std::exp(-logit));
            }
            *app_ms += ms_since(ta);
        }
    }
}

void SignPipeline::classify(const Nv12View &f, std::vector<Detection> &dets) {
    std::lock_guard<std::mutex> lk(cls_mu_);
    double app = 0;
    classify_locked(f, dets, &app);
}

void SignPipeline::view_rect(int view, int fw, int fh, float out[4]) const {
    if (view == 1 && has_view2()) {
        out[0] = std::floor(cfg_.roi2[0] * fw), out[1] = std::floor(cfg_.roi2[1] * fh);
        out[2] = std::ceil(cfg_.roi2[2] * fw), out[3] = std::ceil(cfg_.roi2[3] * fh);
        out[0] = std::max(0.f, out[0]), out[1] = std::max(0.f, out[1]);
        out[2] = std::min(float(fw), std::max(out[0] + 2, out[2])), out[3] = std::min(float(fh), std::max(out[1] + 2, out[3]));
    } else {
        out[0] = out[1] = 0, out[2] = float(fw), out[3] = float(fh);
    }
}

void SignPipeline::process(const Nv12View &f, std::vector<Detection> &out, Timing *tm, int view) {
    out.clear();
    Timing t;
    const auto t0 = Clock::now();

    // --- letterbox (geometry exactly as Ultralytics / docs/SIGN_MODELS.md); the second view is the same
    // letterbox of its window instead of the whole frame
    float vr[4];
    view_rect(view, f.width, f.height, vr);
    const double rx0 = vr[0], ry0 = vr[1], rw = vr[2] - vr[0], rh = vr[3] - vr[1];
    const double r = std::min(double(in_w_) / rw, double(in_h_) / rh);
    const int nw = int(std::lround(rw * r)), nh = int(std::lround(rh * r));
    const int left = int(std::lround((in_w_ - nw) / 2.0 - 0.1)), top = int(std::lround((in_h_ - nh) / 2.0 - 0.1));
    const size_t plane = size_t(in_w_) * in_h_;
    // Padding stays at 114/255 (set once per letterbox geometry); the image area is fully rewritten.
    if (nw != lb_w_ || nh != lb_h_ || left != lb_l_ || top != lb_t_) {
        std::fill(det_in_.begin(), det_in_.end(), 114.f / 255.f);
        lb_w_ = nw;
        lb_h_ = nh;
        lb_l_ = left;
        lb_t_ = top;
    }
    nv12_region_to_rgb_planar(f, rx0, ry0, rw, rh, nw, nh, det_in_.data(), in_w_, plane, left, top);
    const auto t1 = Clock::now();
    t.pre_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --- detector (the GPU-schedule hook runs between recording and submitting, see runner.h)
    if (before_det) {
        det_->run_hooked(det_in_.data(), det_out_.data(), [&] {
            const auto tw0 = Clock::now();
            before_det();
            t.wait_ms = std::chrono::duration<double, std::milli>(Clock::now() - tw0).count();
        });
    } else {
        det_->run(det_in_.data(), det_out_.data());
    }
    if (after_det) after_det();
    const auto t2 = Clock::now();
    t.det_ms = std::chrono::duration<double, std::milli>(t2 - t1).count() - t.wait_ms;

    // --- decode + NMS
    std::vector<Detection> cand;
    const int nc = n_out_ - 4;
    const float *o = det_out_.data();
    const size_t N = size_t(n_anchor_);
    for (size_t i = 0; i < N; ++i) {
        float best = o[4 * N + i];
        for (int c = 1; c < nc; ++c) best = std::max(best, o[(4 + c) * N + i]);
        if (!(best > cfg_.det_thr)) continue;
        const float cx = o[i], cy = o[N + i], w = o[2 * N + i], h = o[3 * N + i];
        Detection d;
        d.x0 = cx - w / 2;
        d.y0 = cy - h / 2;
        d.x1 = cx + w / 2;
        d.y1 = cy + h / 2;
        d.score = best;
        cand.push_back(std::move(d));
    }
    // Single-class detector: plain greedy NMS (with nc > 1 this is class-agnostic,
    // which is what a sign tracker wants anyway).
    std::stable_sort(cand.begin(), cand.end(), [](const Detection &a, const Detection &b) { return a.score > b.score; });
    std::vector<char> dead(cand.size(), 0);
    for (size_t i = 0; i < cand.size() && int(out.size()) < cfg_.max_det; ++i) {
        if (dead[i]) continue;
        for (size_t j = i + 1; j < cand.size(); ++j)
            if (!dead[j] && iou(cand[i], cand[j]) > cfg_.nms_iou) dead[j] = 1;
        Detection d = cand[i];
        const float W = float(f.width), H = float(f.height);
        auto mx = [&](float x) { return std::min(std::max(float(rx0 + (x - left) / r), float(vr[0])), std::min(W, vr[2])); };
        auto my = [&](float y) { return std::min(std::max(float(ry0 + (y - top) / r), float(vr[1])), std::min(H, vr[3])); };
        d.x0 = mx(d.x0);
        d.x1 = mx(d.x1);
        d.y0 = my(d.y0);
        d.y1 = my(d.y1);
        if (d.x1 - d.x0 < cfg_.min_box_px || d.y1 - d.y0 < cfg_.min_box_px) continue;
        if (view == 1 && has_view2()) {  // cut by an inner edge of the window: the whole-frame view owns it
            const float m = 2.f;
            if ((vr[0] > 0 && d.x0 <= vr[0] + m) || (vr[2] < W && d.x1 >= vr[2] - m) || (vr[1] > 0 && d.y0 <= vr[1] + m) ||
                (vr[3] < H && d.y1 >= vr[3] - m))
                continue;
        }
        out.push_back(std::move(d));
    }
    const auto t3 = Clock::now();
    t.post_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // --- classifier per box
    {
        std::lock_guard<std::mutex> lk(cls_mu_);
        classify_locked(f, out, &t.app_ms);
    }
    t.cls_ms = ms_since(t3);
    t.total_ms = ms_since(t0);
    if (tm) *tm = t;
}

}  // namespace signs
}  // namespace roadsense
