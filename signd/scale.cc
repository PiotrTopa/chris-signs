#include "scale.h"

#include <algorithm>
#include <cmath>

#include "lights.h"

namespace roadsense {
namespace signs {

namespace {

inline float bil_u8(const uint8_t *p, int stride, double x, double y) {
    const int x0 = int(x), y0 = int(y);
    const float fx = float(x - x0), fy = float(y - y0);
    const uint8_t *r0 = p + size_t(y0) * stride + x0, *r1 = r0 + stride;
    const float a = r0[0] + fx * (r0[1] - r0[0]);
    const float b = r1[0] + fx * (r1[1] - r1[0]);
    return a + fy * (b - a);
}

inline float bil_f(const float *p, int w, double x, double y) {
    const int x0 = int(x), y0 = int(y);
    const float fx = float(x - x0), fy = float(y - y0);
    const float *r0 = p + size_t(y0) * w + x0, *r1 = r0 + w;
    const float a = r0[0] + fx * (r0[1] - r0[0]);
    const float b = r1[0] + fx * (r1[1] - r1[0]);
    return a + fy * (b - a);
}

// Solves A x = b (n x n, row-major, A symmetric positive definite expected) in place by Gaussian elimination with
// partial pivoting; false when singular.
bool solve(double *A, double *b, int n) {
    for (int c = 0; c < n; ++c) {
        int p = c;
        for (int r = c + 1; r < n; ++r)
            if (std::fabs(A[r * n + c]) > std::fabs(A[p * n + c])) p = r;
        if (std::fabs(A[p * n + c]) < 1e-12) return false;
        if (p != c) {
            for (int k = 0; k < n; ++k) std::swap(A[c * n + k], A[p * n + k]);
            std::swap(b[c], b[p]);
        }
        for (int r = c + 1; r < n; ++r) {
            const double f = A[r * n + c] / A[c * n + c];
            for (int k = c; k < n; ++k) A[r * n + k] -= f * A[c * n + k];
            b[r] -= f * b[c];
        }
    }
    for (int c = n - 1; c >= 0; --c) {
        double s = b[c];
        for (int k = c + 1; k < n; ++k) s -= A[c * n + k] * b[k];
        b[c] = s / A[c * n + c];
    }
    return true;
}

// (A^-1)[0][0] for a 5 x 5 matrix (A copied).
bool inv00(const double *A0, int n, double *out) {
    double A[25], e[5] = {1, 0, 0, 0, 0};
    std::copy(A0, A0 + n * n, A);
    if (!solve(A, e, n)) return false;
    *out = e[0];
    return true;
}

int group_of(int cls) { return (cls == 0 || cls == 1 || cls == 6 || cls == 7) ? 0 : 1; }

float iou_box(const float *a, const float *b) {
    const float ix = std::max(0.f, std::min(a[2], b[2]) - std::max(a[0], b[0]));
    const float iy = std::max(0.f, std::min(a[3], b[3]) - std::max(a[1], b[1]));
    const float i = ix * iy;
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9f);
}

}  // namespace

bool sample_luma(const Nv12View &f, double cx, double cy, double k, int w, int h, std::vector<float> &out) {
    if (w < 1 || h < 1 || k <= 0) return false;
    // area taps per axis when downscaling by more than 1.5 (below that plain bilinear: a 2-tap average at a
    // factor near 1 would blur one image of a pair and not the other, which biases the scale)
    const int n = k < 0.67 ? std::min(4, int(std::ceil(1.0 / k - 1e-6))) : 1;
    const double inv = 1.0 / k;
    const double ext = n > 1 ? 0.5 * inv : 0.0;
    const double xa = cx - 0.5 * (w - 1) * inv - ext, xb = cx + 0.5 * (w - 1) * inv + ext;
    const double ya = cy - 0.5 * (h - 1) * inv - ext, yb = cy + 0.5 * (h - 1) * inv + ext;
    if (xa < 0 || ya < 0 || xb > f.width - 1.001 || yb > f.height - 1.001) return false;
    out.resize(size_t(w) * h);
    double off[4];
    for (int i = 0; i < n; ++i) off[i] = n > 1 ? ((i + 0.5) / n - 0.5) * inv : 0.0;
    const float norm = 1.f / float(n * n);
    for (int r = 0; r < h; ++r) {
        const double y = cy + (r - 0.5 * (h - 1)) * inv;
        for (int c = 0; c < w; ++c) {
            const double x = cx + (c - 0.5 * (w - 1)) * inv;
            float s = 0;
            for (int a = 0; a < n; ++a)
                for (int b = 0; b < n; ++b) s += bil_u8(f.y, f.y_stride, x + off[b], y + off[a]);
            out[size_t(r) * w + c] = s * norm;
        }
    }
    return true;
}

ScaleResult measure_scale(const std::vector<float> &tpl, int tw, int th, double k0, const Nv12View &f, double prior_ln_s,
                          double pcx, double pcy, double half_range, int shift_r, const ScaleConfig &cfg) {
    ScaleResult res;
    const int N = tw * th;
    if (N < 64 || std::min(tw, th) < 8 || int(tpl.size()) != N) return res.why = kSmall, res;
    double mT = 0, vT = 0;
    for (float v : tpl) mT += v;
    mT /= N;
    for (float v : tpl) vT += (v - mT) * (v - mT);
    if (std::sqrt(vT / N) < cfg.min_tex) return res.why = kTexture, res;  // textureless
    std::vector<float> Tn(static_cast<size_t>(N));
    const double nT = 1.0 / std::sqrt(vT);
    for (int i = 0; i < N; ++i) Tn[size_t(i)] = float((tpl[size_t(i)] - mT) * nT);
    const double tcx = 0.5 * (tw - 1), tcy = 0.5 * (th - 1);

    // Search image around the predicted centre. A native template (k0 = 1, raw pixels) is searched in a raw crop of
    // the frame (integer-aligned, k1 = 1): the warp then interpolates once, like nothing else in the pair, and no
    // blur differs between the two images (a resampled search image is blurrier than a raw template, which biases
    // the scale). A downscaled template (area-averaged) is searched in an image area-averaged at the prior's scale,
    // so the object appears about as large as in the template. The warp's scale is a = s * k1 / k0.
    const bool native = k0 >= 0.999;
    const double k1 = native ? 1.0 : std::min(0.66, k0 * std::exp(-prior_ln_s));
    const double lk = std::log(k1 / k0), gc = prior_ln_s + lk;  // grid centre in ln a
    if (native) pcx = std::round(pcx), pcy = std::round(pcy);
    const int R = shift_r;
    const double amax = std::exp(std::fabs(gc) + half_range) * 1.02;
    const int hx = int(std::ceil(amax * (tcx + R))) + 2, hy = int(std::ceil(amax * (tcy + R))) + 2;
    const int IW = 2 * hx + 1, IH = 2 * hy + 1;
    std::vector<float> I;
    if (!sample_luma(f, pcx, pcy, k1, IW, IH, I)) return res.why = kFrameEdge, res;  // leaves the frame

    // NCC grid: log scale x integer shift (template units; shift in search-image pixels = a * shift). Scale step:
    // 1 / template radius (the fit converges from a peak whose scale error moves the template's edge by ~1 px),
    // 0.04 .. 0.08. All shifts at the prior's scale (the centre does not move with the scale), then +-2 around that
    // peak at the other scales.
    const double rmax = std::max(tcx, tcy);
    const double step = std::min(half_range / 2.0, std::min(0.08, std::max(0.04, 1.0 / std::max(1.0, rmax))));
    const int nh = std::min(7, int(std::ceil(half_range / step - 1e-9)));
    const int EW = tw + 2 * R, EH = th + 2 * R;
    std::vector<float> Ia(size_t(EW) * EH);
    double best = -2;
    int bth = 0, bdx = 0, bdy = 0;
    auto scan = [&](int it, int x0, int x1, int y0, int y1) {
        const double a = std::exp(gc + it * step);
        for (int r = y0 + R; r < y1 - 1 + R + th; ++r)  // shifts y0 .. y1-1 read these rows
            for (int c = x0 + R; c < x1 - 1 + R + tw; ++c)
                Ia[size_t(r) * EW + c] = bil_f(I.data(), IW, hx + a * (c - R - tcx), hy + a * (r - R - tcy));
        for (int dy = y0; dy < y1; ++dy)
            for (int dx = x0; dx < x1; ++dx) {
                double sI = 0, sII = 0, sTI = 0;
                for (int r = 0; r < th; ++r) {
                    const float *ip = &Ia[size_t(r + R + dy) * EW + R + dx];
                    const float *tp = &Tn[size_t(r) * tw];
                    float a1 = 0, a2 = 0, a3 = 0;
                    for (int c = 0; c < tw; ++c) {
                        const float v = ip[c];
                        a1 += v, a2 += v * v, a3 += tp[c] * v;
                    }
                    sI += a1, sII += a2, sTI += a3;
                }
                const double den = sII - sI * sI / N;
                if (den <= 1e-6) continue;
                const double ncc = sTI / std::sqrt(den);
                if (ncc > best) best = ncc, bth = it, bdx = dx, bdy = dy;
            }
    };
    scan(0, -R, R + 1, -R, R + 1);
    const int sx = bdx, sy = bdy;
    for (int it = -nh; it <= nh; ++it)
        if (it) scan(it, std::max(-R, sx - 2), std::min(R, sx + 2) + 1, std::max(-R, sy - 2), std::min(R, sy + 2) + 1);
    res.ncc = float(best);
    if (best < cfg.min_ncc) return res.why = kNcc, res;
    if (std::abs(bth) == nh || std::abs(bdx) == R || std::abs(bdy) == R) return res.why = kSearchEdge, res;  // peak at the search edge

    // Gauss-Newton: T(u) ~ g * I(a u + d) + b, parameters (ln a, dx, dy, g, b)
    std::vector<float> Gx(I.size()), Gy(I.size());
    for (int r = 0; r < IH; ++r)
        for (int c = 0; c < IW; ++c) {
            const size_t i = size_t(r) * IW + c;
            const int cl = std::max(0, c - 1), cr = std::min(IW - 1, c + 1), ru = std::max(0, r - 1), rd = std::min(IH - 1, r + 1);
            Gx[i] = (I[size_t(r) * IW + cr] - I[size_t(r) * IW + cl]) / float(cr - cl);
            Gy[i] = (I[size_t(rd) * IW + c] - I[size_t(ru) * IW + c]) / float(rd - ru);
        }
    const double th0 = gc + bth * step;
    double p[5] = {th0, std::exp(th0) * bdx, std::exp(th0) * bdy, 1.0, 0.0};
    {   // gain and offset at the start
        const double a = std::exp(p[0]);
        double s1 = 0, s2 = 0, st = 0, sit = 0;
        for (int r = 0; r < th; ++r)
            for (int c = 0; c < tw; ++c) {
                const double v = bil_f(I.data(), IW, hx + a * (c - tcx) + p[1], hy + a * (r - tcy) + p[2]);
                const double t = tpl[size_t(r) * tw + c];
                s1 += v, s2 += v * v, st += t, sit += v * t;
            }
        const double d = N * s2 - s1 * s1;
        if (d > 1e-9) p[3] = (N * sit - s1 * st) / d, p[4] = (st - p[3] * s1) / N;
    }
    double H[25], gvec[5], ssr = 0;
    std::vector<float> rres(static_cast<size_t>(N));
    bool conv = false;
    for (int iter = 0; iter < 25; ++iter) {
        std::fill(H, H + 25, 0.0);
        std::fill(gvec, gvec + 5, 0.0);
        ssr = 0;
        const double a = std::exp(p[0]);
        for (int r = 0; r < th; ++r) {
            const double uy = r - tcy;
            for (int c = 0; c < tw; ++c) {
                const double ux = c - tcx;
                const double x = hx + a * ux + p[1], y = hy + a * uy + p[2];
                if (x < 0 || y < 0 || x > IW - 1.001 || y > IH - 1.001) return res.why = kFit, res;
                const double v = bil_f(I.data(), IW, x, y), gx = bil_f(Gx.data(), IW, x, y), gy = bil_f(Gy.data(), IW, x, y);
                const double rr = p[3] * v + p[4] - tpl[size_t(r) * tw + c];
                const double J[5] = {p[3] * a * (gx * ux + gy * uy), p[3] * gx, p[3] * gy, v, 1.0};
                for (int m = 0; m < 5; ++m) {
                    gvec[m] += J[m] * rr;
                    for (int n = m; n < 5; ++n) H[m * 5 + n] += J[m] * J[n];
                }
                ssr += rr * rr;
                rres[size_t(r) * tw + c] = float(rr);
            }
        }
        for (int m = 0; m < 5; ++m)
            for (int n = 0; n < m; ++n) H[m * 5 + n] = H[n * 5 + m];
        if (conv) break;  // H, residuals at the converged parameters
        double A[25], d[5];
        std::copy(H, H + 25, A);
        for (int m = 0; m < 5; ++m) A[m * 5 + m] *= 1.0 + 1e-6, d[m] = -gvec[m];
        if (!solve(A, d, 5)) return res.why = kFit, res;
        for (int m = 0; m < 5; ++m) p[m] += d[m];
        // Converged when the step is small against the parameters' own uncertainty (bilinear interpolation makes
        // the cost piecewise smooth, so Gauss-Newton can dither at the 1e-3 px level forever)
        double hi00;
        const double s2i = ssr / std::max(1, N - 5);
        const double se0 = inv00(H, 5, &hi00) && hi00 > 0 ? std::sqrt(s2i * hi00) : 0.0;
        const double sed = std::sqrt(s2i / std::max(1e-12, std::min(H[6], H[12])));
        if (std::fabs(d[0]) < std::max(1e-5, 0.05 * se0) && std::fabs(d[1]) < std::max(1e-3, 0.05 * sed) &&
            std::fabs(d[2]) < std::max(1e-3, 0.05 * sed))
            conv = true;
        if (std::fabs(p[0] - th0) > 2 * step || std::fabs(p[1] - std::exp(th0) * bdx) > 2.0 ||
            std::fabs(p[2] - std::exp(th0) * bdy) > 2.0 || p[3] <= 0.05)
            return res.why = kRunaway, res;  // ran away from the grid peak
    }
    if (!conv) return res.why = kNoConv, res;
    res.r2 = float(1.0 - ssr / vT);
    if (res.r2 < cfg.min_r2) return res.why = kR2, res;
    const double s2 = ssr / std::max(1, N - 5);
    // The image gradients in J carry the image's noise too (errors in variables): their noise adds
    // E[dJ^T dJ] to J^T J and the plain inverse understates the variance. Remove that expectation from the geometric
    // block: noise per image sigma_I^2 = s2 / (1 + g^2) (the residual holds both images' noise), gradient noise
    // sigma_I^2 / 2 per axis (central difference); at most half of any diagonal term.
    {
        double suu = 0;
        for (int r = 0; r < th; ++r)
            for (int c = 0; c < tw; ++c) suu += (c - tcx) * (c - tcx) + (r - tcy) * (r - tcy);
        const double g2 = p[3] * p[3], a = std::exp(p[0]), sg2 = s2 / (1.0 + g2) * 0.5;
        const double d0 = g2 * a * a * sg2 * suu, d1 = g2 * sg2 * N;
        const double lim = std::min({1.0, 0.5 * H[0] / std::max(d0, 1e-12), 0.5 * H[6] / std::max(d1, 1e-12),
                                     0.5 * H[12] / std::max(d1, 1e-12)});
        H[0] -= lim * d0, H[6] -= lim * d1, H[12] -= lim * d1;
    }
    double h00;
    if (!inv00(H, 5, &h00) || h00 <= 0) return res.why = kFit, res;
    // Residuals of neighbouring pixels are correlated (resampling, optics, JPEG): inflate by the lag-1
    // autocorrelation along each axis, (1+rho)/(1-rho) per axis.
    double cx = 0, cy = 0, c0 = 0;
    for (int r = 0; r < th; ++r)
        for (int c = 0; c < tw; ++c) {
            const double v = rres[size_t(r) * tw + c];
            c0 += v * v;
            if (c + 1 < tw) cx += v * rres[size_t(r) * tw + c + 1];
            if (r + 1 < th) cy += v * rres[size_t(r + 1) * tw + c];
        }
    double rx = c0 > 0 ? cx / c0 * double(N) / double(N - th) : 0, ry = c0 > 0 ? cy / c0 * double(N) / double(N - tw) : 0;
    rx = std::min(0.9, std::max(0.0, rx)), ry = std::min(0.9, std::max(0.0, ry));
    const double infl = (1 + rx) / (1 - rx) * (1 + ry) / (1 - ry);
    res.ln_s = p[0] - lk;
    res.ln_s_std = std::sqrt(s2 * h00 * infl);
    res.ok = res.ln_s_std <= cfg.max_std;
    res.why = res.ok ? kMeasured : kStd;
    return res;
}

const char *scale_why_name(int why) {
    static const char *n[kNumScaleWhy] = {"measured", "tiny", "textureless", "frame_edge", "ncc", "search_edge", "fit", "runaway", "no_convergence",
                                          "r2", "std", "over_cap", "box_at_edge", "new_object", "no_baseline", "clipped", "lamps_only"};
    return why >= 0 && why < kNumScaleWhy ? n[why] : "?";
}

void ScaleRateEstimator::update(const Nv12View &f, int64_t ts, std::vector<RoadUserDet> &users, std::vector<RoadUserDet> *low) {
    for (RoadUserDet &u : users) u.scale_ok = false, u.scale_rate = u.scale_rate_std = u.scale_dt_s = 0.f, u.scale_src = kSrcNone;
    if (low)
        for (RoadUserDet &u : *low) u.scale_ok = false, u.scale_rate = u.scale_rate_std = u.scale_dt_s = 0.f, u.scale_src = kSrcNone;
    if (ts < last_ts_) chains_.clear();  // time went backwards (new ring / replay restart)
    last_ts_ = ts;
    const int64_t gap = int64_t(cfg_.chain_max_gap_s * 1e9), keep = int64_t((cfg_.max_dt_s + 0.3) * 1e9);
    {
        std::vector<Chain> live;
        for (Chain &c : chains_) {
            if (c.e.empty() || ts - c.e.back().ts > gap) continue;
            while (c.e.size() > 1 && ts - c.e.front().ts > keep) c.e.erase(c.e.begin());
            live.push_back(std::move(c));
        }
        chains_.swap(live);
    }
    // the boxes measured: largest first; then the low-score boxes (largest first) while the cap has room
    std::vector<RoadUserDet *> bx;
    const auto by_area = [](const RoadUserDet *a, const RoadUserDet *b) {
        return (a->x1 - a->x0) * (a->y1 - a->y0) > (b->x1 - b->x0) * (b->y1 - b->y0);
    };
    for (RoadUserDet &u : users)
        if (u.x1 > u.x0 && u.y1 > u.y0) bx.push_back(&u);
    std::stable_sort(bx.begin(), bx.end(), by_area);
    if (int(bx.size()) > cfg_.max_boxes) why_[kOverCap] += bx.size() - size_t(cfg_.max_boxes), bx.resize(size_t(cfg_.max_boxes));
    const size_t n_high = bx.size();
    if (low) {
        for (RoadUserDet &u : *low)
            if (u.x1 > u.x0 && u.y1 > u.y0) bx.push_back(&u);
        std::stable_sort(bx.begin() + long(n_high), bx.end(), by_area);
        if (int(bx.size()) > cfg_.max_boxes) bx.resize(size_t(cfg_.max_boxes));
    }

    // association: each box to the chain whose predicted box overlaps it best; the first list's boxes first, then
    // the low-score boxes among the chains still free
    struct P { float s; size_t b, c; };
    std::vector<int> match(bx.size(), -1);
    std::vector<char> used(chains_.size(), 0);
    for (int stage = 0; stage < 2; ++stage) {
        std::vector<P> pairs;
        for (size_t bi = stage ? n_high : 0; bi < (stage ? bx.size() : n_high); ++bi) {
            const RoadUserDet &u = *bx[bi];
            const float ub[4] = {u.x0, u.y0, u.x1, u.y1};
            for (size_t ci = 0; ci < chains_.size(); ++ci) {
                const Chain &c = chains_[ci];
                if (used[ci] || c.grp != group_of(u.cls)) continue;
                const Entry &n = c.e.back();
                double vx = 0, vy = 0, vs = 0;
                for (size_t j = c.e.size() - 1; j-- > 0;) {  // velocity from the newest entry at least 0.15 s older
                    const Entry &o = c.e[j];
                    const double dt = double(n.ts - o.ts) * 1e-9;
                    if (dt < 0.15) continue;
                    vx = ((n.box[0] + n.box[2]) - (o.box[0] + o.box[2])) * 0.5 / dt;
                    vy = ((n.box[1] + n.box[3]) - (o.box[1] + o.box[3])) * 0.5 / dt;
                    vs = std::log(std::max(1e-3, double(n.box[2] - n.box[0]) * (n.box[3] - n.box[1]) /
                                                    std::max(1e-3, double(o.box[2] - o.box[0]) * (o.box[3] - o.box[1])))) * 0.5 / dt;
                    vs = std::max(-2.0, std::min(2.0, vs));
                    break;
                }
                const double dt = double(ts - n.ts) * 1e-9, g = std::exp(vs * dt);
                const double cx = (n.box[0] + n.box[2]) * 0.5 + vx * dt, cy = (n.box[1] + n.box[3]) * 0.5 + vy * dt;
                const double w = (n.box[2] - n.box[0]) * 0.5 * g, h = (n.box[3] - n.box[1]) * 0.5 * g;
                const float pb[4] = {float(cx - w), float(cy - h), float(cx + w), float(cy + h)};
                const float s = iou_box(ub, pb);
                if (s >= cfg_.match_iou) pairs.push_back({s, bi, ci});
            }
        }
        std::sort(pairs.begin(), pairs.end(), [](const P &a, const P &b) { return a.s > b.s; });
        for (const P &p : pairs)
            if (match[p.b] < 0 && !used[p.c]) match[p.b] = int(p.c), used[p.c] = 1;
    }

    for (size_t bi = 0; bi < bx.size(); ++bi) {
        RoadUserDet &u = *bx[bi];
        const bool is_low = bi >= n_high;
        if (is_low && match[bi] < 0) continue;  // a low-score box never starts a chain
        Entry e;
        e.ts = ts;
        e.box[0] = u.x0, e.box[1] = u.y0, e.box[2] = u.x1, e.box[3] = u.y1;
        e.cx = (u.x0 + u.x1) * 0.5, e.cy = (u.y0 + u.y1) * 0.5;
        e.edge = u.x0 <= 1.f || u.y0 <= 1.f || u.x1 >= f.width - 1.f || u.y1 >= f.height - 1.f;
        e.clip = u.clip;
        e.low = is_low;
        if (cfg_.lamps && u.lamps.ok) e.lamp = true, e.lsep = u.lamps.sep, e.lsep_std = u.lamps.sep_std, e.lcol = u.lamps.colour;
        const double iw = (u.x1 - u.x0) * cfg_.inner_frac, ih = (u.y1 - u.y0) * cfg_.inner_frac;
        // about patch_px^2 template pixels whatever the aspect (a fixed long side would leave a flat box, e.g. a near
        // car cut by the view's edge, two rows high and its vertical shift undetermined); the long side may grow to
        // 4 patch_px so that such a strip keeps its 8 rows (measure_scale() refuses fewer): native pixels while that
        // needs at most a 1/0.85 reduction (raw, integer-aligned: no interpolation blur), larger regions
        // area-averaged down to at most 0.66 (measure_scale() keeps the two kinds apart)
        const double kr = std::min(4.0 * cfg_.patch_px / std::max(iw, ih),
                                   std::max(cfg_.patch_px / std::sqrt(iw * ih), 8.5 / std::min(iw, ih)));
        e.k = kr >= 0.85 ? 1.0 : std::min(kr, 0.66);
        e.w = std::max(1, int(std::lround(iw * e.k))), e.h = std::max(1, int(std::lround(ih * e.k)));
        if (e.k >= 1.0) e.cx = std::round(e.cx - 0.5 * (e.w - 1)) + 0.5 * (e.w - 1), e.cy = std::round(e.cy - 0.5 * (e.h - 1)) + 0.5 * (e.h - 1);
        if (!e.edge && std::min(iw, ih) >= cfg_.min_side_px) {
            // the template covers exactly e.w / e.k x e.h / e.k frame px around the box centre
            if (!sample_luma(f, e.cx, e.cy, e.k, e.w, e.h, e.tpl)) e.tpl.clear();
        }
        int why = match[bi] < 0 ? kNew : e.edge ? kBoxEdge : kNoBaseline;
        // ---- patch alignment against one earlier patch of the chain
        const auto align = [&](const Entry *t, int *why_out) {
            ScaleResult r;
            const double w0 = t->box[2] - t->box[0], h0 = t->box[3] - t->box[1], w1 = u.x1 - u.x0, h1 = u.y1 - u.y0;
            const int cl = t->clip | u.clip;
            const bool hx_ok = !(cl & (1 | 4)), hy_ok = !(cl & (2 | 8));
            double prior = 0;
            bool ok = true;
            if (hx_ok && hy_ok) prior = 0.5 * (std::log(w1 / w0) + std::log(h1 / h0));
            else if (hx_ok) prior = std::log(w1 / w0);
            else if (hy_ok) prior = std::log(h1 / h0);
            else ok = false;
            // predicted position of the template centre: box centre, or anchored on the free edge of a clipped axis
            const double sp = std::exp(prior);
            double pcx = e.cx, pcy = e.cy;
            if ((cl & 1) && (cl & 4)) ok = false;
            else if (cl & 1) pcx = u.x1 - sp * (t->box[2] - t->cx);
            else if (cl & 4) pcx = u.x0 + sp * (t->cx - t->box[0]);
            if ((cl & 2) && (cl & 8)) ok = false;
            else if (cl & 2) pcy = u.y1 - sp * (t->box[3] - t->cy);
            else if (cl & 8) pcy = u.y0 + sp * (t->cy - t->box[1]);
            if (!ok) return *why_out = kClipped, r;
            // search ranges from the box jitter: edges ~1.5 px + 3 % of the size each, in both frames
            const double side = std::sqrt(std::min(w0, w1) * std::min(h0, h1));
            const double sig_ln = (2.0 * (1.5 + 0.03 * side) / side) / ((hx_ok && hy_ok) ? std::sqrt(2.0) : 1.0);
            const double half = std::min(0.28, std::max(0.08, 3.0 * sig_ln));
            const double k1 = t->k >= 0.999 ? 1.0 : std::min(0.66, t->k * std::exp(-prior));  // as measure_scale()
            const double sig_c = 1.5 + 0.03 * std::max(w1, h1);
            const int R = std::min(6, std::max(2, int(std::ceil(3.0 * sig_c * k1))));
            r = measure_scale(t->tpl, t->w, t->h, t->k, f, prior, pcx, pcy, half, R, cfg_);
            r.dt_s = double(ts - t->ts) * 1e-9;
            *why_out = r.why;
            return r;
        };
        bool p_ok = false;
        double p_rate = 0, p_std = 0, p_dt = 0;
        if (match[bi] >= 0 && !e.edge && cfg_.patch) {
            const Chain &c = chains_[size_t(match[bi])];
            std::vector<const Entry *> tp;  // usable entries in the baseline window, oldest first
            for (const Entry &o : c.e) {
                const double dt = double(ts - o.ts) * 1e-9;
                if (dt > cfg_.max_dt_s || dt < cfg_.min_dt_s || o.tpl.empty() || o.edge) continue;
                tp.push_back(&o);
            }
            // A low-score box is a poor template (its edges are 2-3 times as far off, so the scale prior and the
            // search window sit wrong): where the window holds first-list patches, only those are baselines.
            // (Measured: with low boxes as templates the a1-night-fog ride lost 11 of 162 values.)
            if (std::any_of(tp.begin(), tp.end(), [](const Entry *o) { return !o->low; }))
                tp.erase(std::remove_if(tp.begin(), tp.end(), [](const Entry *o) { return o->low; }), tp.end());
            if (!tp.empty()) {
                const ScaleResult r0 = align(tp[0], &why);
                if (cfg_.baselines <= 1 || tp.size() == 1) {
                    if (r0.ok) p_ok = true, p_rate = r0.ln_s / r0.dt_s, p_std = r0.ln_s_std / r0.dt_s, p_dt = r0.dt_s;
                } else {
                    // the oldest and up to baselines - 1 others, spread over the window
                    std::vector<ScaleResult> rs;
                    if (r0.ok) rs.push_back(r0);
                    const int extra = std::min(cfg_.baselines - 1, int(tp.size()) - 1);
                    for (int k = 1; k <= extra; ++k) {
                        const size_t j = size_t(std::lround(double(k) * double(tp.size() - 1) / double(extra)));
                        int w2;
                        const ScaleResult r = align(tp[j], &w2);
                        if (r.ok) rs.push_back(r);
                    }
                    // generalised least squares of ln_s_i = rate * dt_i, cov(i, j) = rho * std_i * std_j (i != j)
                    const auto gls = [&](const std::vector<ScaleResult> &v, double *rate, double *var, double *chi2, size_t *worst) {
                        const size_t n = v.size();
                        std::vector<double> C(n * n), x(n), y(n);
                        const auto solve_c = [&](std::vector<double> rhs) {
                            std::vector<double> A(C);
                            if (!solve(A.data(), rhs.data(), int(n))) rhs.assign(n, 0.0);
                            return rhs;
                        };
                        for (size_t i = 0; i < n; ++i)
                            for (size_t j = 0; j < n; ++j) C[i * n + j] = (i == j ? 1.0 : cfg_.multi_rho) * v[i].ln_s_std * v[j].ln_s_std;
                        for (size_t i = 0; i < n; ++i) x[i] = v[i].dt_s, y[i] = v[i].ln_s;
                        const std::vector<double> cx = solve_c(x);
                        double xcx = 0, xcy = 0;
                        for (size_t i = 0; i < n; ++i) xcx += x[i] * cx[i], xcy += y[i] * cx[i];
                        if (!(xcx > 0)) return false;
                        *rate = xcy / xcx, *var = 1.0 / xcx;
                        std::vector<double> res(n);
                        for (size_t i = 0; i < n; ++i) res[i] = y[i] - *rate * x[i];
                        const std::vector<double> cr = solve_c(res);
                        *chi2 = 0, *worst = 0;
                        double wz = -1;
                        for (size_t i = 0; i < n; ++i) {
                            *chi2 += res[i] * cr[i];
                            const double z = std::fabs(res[i]) / v[i].ln_s_std;
                            if (z > wz) wz = z, *worst = i;
                        }
                        return true;
                    };
                    double rate = 0, var = 0, chi2 = 0;
                    size_t worst = 0;
                    if (rs.size() >= 3 && gls(rs, &rate, &var, &chi2, &worst) &&
                        std::fabs(rs[worst].ln_s - rate * rs[worst].dt_s) > cfg_.multi_out * rs[worst].ln_s_std)
                        rs.erase(rs.begin() + long(worst));  // one baseline against the others: left out
                    if (rs.size() == 1) {
                        p_ok = true, p_rate = rs[0].ln_s / rs[0].dt_s, p_std = rs[0].ln_s_std / rs[0].dt_s, p_dt = rs[0].dt_s;
                    } else if (rs.size() >= 2 && gls(rs, &rate, &var, &chi2, &worst)) {
                        const double infl = std::max(1.0, chi2 / double(rs.size() - 1));
                        p_ok = true, p_rate = rate, p_std = std::sqrt(var * infl);
                        for (const ScaleResult &r : rs) p_dt = std::max(p_dt, r.dt_s);
                    }
                    if (p_ok) why = kMeasured;
                }
            }
        }
        // ---- lamp pair: weighted line through ln(separation) over the chain's passes of the window
        bool l_ok = false;
        double l_rate = 0, l_std = 0, l_dt = 0;
        if (cfg_.lamps && e.lamp && e.lsep >= cfg_.lamp_min_sep && match[bi] >= 0) {
            const Chain &c = chains_[size_t(match[bi])];
            std::vector<double> t{0.0}, y{std::log(double(e.lsep))}, sg{double(e.lsep_std) / e.lsep};
            double last = y[0];
            for (size_t j = c.e.size(); j-- > 0;) {
                const Entry &o = c.e[j];
                const double dt = double(ts - o.ts) * 1e-9;
                if (dt > cfg_.max_dt_s) break;
                if (!o.lamp || o.lsep < cfg_.lamp_min_sep) continue;
                const double v = std::log(double(o.lsep));
                if (std::fabs(v - last) > cfg_.lamp_jump || (o.lcol && e.lcol && o.lcol != e.lcol)) break;  // another pair
                t.push_back(-dt), y.push_back(v), sg.push_back(double(o.lsep_std) / o.lsep), last = v;
            }
            const size_t n = t.size();
            if (int(n) >= cfg_.lamp_min_n && -t.back() >= cfg_.min_dt_s) {
                // weighted line, Huber weights re-estimated three times (a pass on another pair of blobs, a lamp
                // coming on: one point off the line must not turn it)
                std::vector<double> hw(n, 1.0);
                double slope = 0, icpt = 0, den = 0, sw = 0;
                bool ok = true;
                for (int it = 0; it < 4 && ok; ++it) {
                    double st = 0, sy = 0, stt = 0, sty = 0;
                    sw = 0;
                    for (size_t i = 0; i < n; ++i) {
                        const double w = hw[i] / (sg[i] * sg[i]);
                        sw += w, st += w * t[i], sy += w * y[i], stt += w * t[i] * t[i], sty += w * t[i] * y[i];
                    }
                    den = sw * stt - st * st;
                    if (!(den > 1e-12)) {
                        ok = false;
                        break;
                    }
                    slope = (sw * sty - st * sy) / den, icpt = (sy - slope * st) / sw;
                    for (size_t i = 0; i < n; ++i) {
                        const double z = std::fabs(y[i] - icpt - slope * t[i]) / sg[i];
                        hw[i] = z <= cfg_.lamp_huber ? 1.0 : cfg_.lamp_huber / z;
                    }
                }
                if (ok) {
                    // the fit's own scatter (all points at full weight) grows the stated std, never shrinks it
                    double chi2 = 0;
                    for (size_t i = 0; i < n; ++i) {
                        const double r = (y[i] - icpt - slope * t[i]) / sg[i];
                        chi2 += r * r;
                    }
                    const double infl = std::max(1.0, chi2 / double(n - 2));
                    l_rate = slope, l_std = std::sqrt(sw / den * infl), l_dt = -t.back();
                    l_ok = l_std <= cfg_.lamp_max_std;
                }
            }
        }
        if (p_ok && l_ok) {
            // both: inverse-variance mean; the two share pixels (the lamps lie in the patch), so its std is not taken
            // below 0.7 of the better one's; where they disagree, half the difference is added
            const double wp = 1.0 / (p_std * p_std), wl = 1.0 / (l_std * l_std), d = p_rate - l_rate;
            double sd = std::max(std::sqrt(1.0 / (wp + wl)), 0.7 * std::min(p_std, l_std));
            if (std::fabs(d) > cfg_.agree_sigma * std::hypot(p_std, l_std)) sd = std::hypot(sd, 0.5 * d);
            u.scale_ok = true, u.scale_src = kSrcBoth;
            u.scale_rate = float((wp * p_rate + wl * l_rate) / (wp + wl)), u.scale_rate_std = float(sd);
            u.scale_dt_s = float((wp * p_dt + wl * l_dt) / (wp + wl));
        } else if (p_ok) {
            u.scale_ok = true, u.scale_src = kSrcPatch;
            u.scale_rate = float(p_rate), u.scale_rate_std = float(p_std), u.scale_dt_s = float(p_dt);
        } else if (l_ok) {
            u.scale_ok = true, u.scale_src = kSrcLamps;
            u.scale_rate = float(l_rate), u.scale_rate_std = float(l_std * cfg_.lamp_only_std_x), u.scale_dt_s = float(l_dt);
            ++why_[kLampOnly];
        }
        ++why_[why];
        if (match[bi] >= 0) {
            chains_[size_t(match[bi])].e.push_back(std::move(e));
        } else {
            Chain c;
            c.grp = group_of(u.cls);
            c.e.push_back(std::move(e));
            chains_.push_back(std::move(c));
        }
    }
    if (chains_.size() > 64) chains_.erase(chains_.begin(), chains_.begin() + long(chains_.size() - 64));
}

}  // namespace signs
}  // namespace roadsense
