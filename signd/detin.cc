#include "detin.h"

#include <algorithm>
#include <cmath>

#include "lights.h"

namespace roadsense {
namespace signs {

// ------------------------------------------------------------------ flags ---

int ru_nearer(const RoadUserDet &a, const RoadUserDet &b, const RuFlagsConfig &cfg) {
    const bool ca = (a.clip | a.trunc) & 8, cb = (b.clip | b.trunc) & 8;
    if (ca && cb) return 0;
    if (ca) return 1;
    if (cb) return -1;
    const float tie = std::max(cfg.tie_px, cfg.tie_rel * std::max(a.y1 - a.y0, b.y1 - b.y0));
    const float d = a.y1 - b.y1;
    return d > tie ? 1 : d < -tie ? -1 : 0;
}

namespace {

void flag_one(RoadUserDet &u, int self, const std::vector<RoadUserDet> &occluders, const RuFlagsConfig &cfg) {
    u.occ = 0.f, u.occ_by = -1, u.occ_edges = 0;
    const float w = u.x1 - u.x0, h = u.y1 - u.y0;
    if (!(w > 0) || !(h > 0)) return;
    constexpr int G = 16;
    uint8_t cov[G][G] = {};
    float best = 0.f;
    bool any = false;
    for (size_t j = 0; j < occluders.size(); ++j) {
        if (int(j) == self) continue;
        const RoadUserDet &o = occluders[j];
        const float ix0 = std::max(u.x0, o.x0), iy0 = std::max(u.y0, o.y0), ix1 = std::min(u.x1, o.x1), iy1 = std::min(u.y1, o.y1);
        if (ix1 <= ix0 || iy1 <= iy0) continue;
        if (ru_nearer(o, u, cfg) < 0) continue;  // u is the nearer one: o does not cover it
        any = true;
        const float a = (ix1 - ix0) * (iy1 - iy0);
        if (a > best) best = a, u.occ_by = int(j);
        // cells whose centre lies in the intersection
        const int c0 = std::max(0, int(std::ceil((ix0 - u.x0) / w * G - 0.5f))), c1 = std::min(G - 1, int(std::floor((ix1 - u.x0) / w * G - 0.5f)));
        const int r0 = std::max(0, int(std::ceil((iy0 - u.y0) / h * G - 0.5f))), r1 = std::min(G - 1, int(std::floor((iy1 - u.y0) / h * G - 0.5f)));
        for (int r = r0; r <= r1; ++r)
            for (int c = c0; c <= c1; ++c) cov[r][c] = 1;
    }
    if (!any) return;
    int n = 0, el = 0, et = 0, er = 0, eb = 0;
    for (int r = 0; r < G; ++r)
        for (int c = 0; c < G; ++c) n += cov[r][c];
    for (int i = 0; i < G; ++i) el += cov[i][0], er += cov[i][G - 1], et += cov[0][i], eb += cov[G - 1][i];
    u.occ = float(n) / float(G * G);
    const int need = int(std::ceil(cfg.edge_hidden * G));
    u.occ_edges = (el >= need ? 1 : 0) | (et >= need ? 2 : 0) | (er >= need ? 4 : 0) | (eb >= need ? 8 : 0);
    if (n == 0) u.occ_by = -1;  // a sliver thinner than a cell
}

int trunc_of(const RoadUserDet &u, int fw, int fh, float e) {
    return (u.x0 <= e ? 1 : 0) | (u.y0 <= e ? 2 : 0) | (u.x1 >= float(fw) - e ? 4 : 0) | (u.y1 >= float(fh) - e ? 8 : 0);
}

}  // namespace

void ru_flags(std::vector<RoadUserDet> &users, std::vector<RoadUserDet> *low, int fw, int fh, const RuFlagsConfig &cfg) {
    for (RoadUserDet &u : users) u.flags = true, u.trunc = trunc_of(u, fw, fh, cfg.edge_px);
    if (low)
        for (RoadUserDet &u : *low) u.flags = true, u.trunc = trunc_of(u, fw, fh, cfg.edge_px);
    for (size_t i = 0; i < users.size(); ++i) flag_one(users[i], int(i), users, cfg);
    if (low)
        for (RoadUserDet &u : *low) flag_one(u, -1, users, cfg);
}

// -------------------------------------------------------------- view cues ---

namespace {

// The box as a working crop: luma box-averaged by the integer factor k, chroma (Cb, Cr) at the block centres.
struct Crop {
    int w = 0, h = 0, k = 1, x0 = 0, y0 = 0;
    std::vector<float> y;
    std::vector<uint8_t> cb, cr;
    float fx(float c) const { return float(x0) + c * k + 0.5f * (k - 1); }  // working column -> frame px
    float fy(float r) const { return float(y0) + r * k + 0.5f * (k - 1); }
};

bool crop_box(const Nv12View &f, const RoadUserDet &u, int max_w, int max_h, bool chroma, Crop &c) {
    const int x0 = std::max(0, int(std::floor(u.x0))), y0 = std::max(0, int(std::floor(u.y0)));
    const int x1 = std::min(f.width, int(std::ceil(u.x1))), y1 = std::min(f.height, int(std::ceil(u.y1)));
    const int bw = x1 - x0, bh = y1 - y0;
    if (bw < 4 || bh < 4) return false;
    c.k = std::max(1, std::max((bw + max_w - 1) / max_w, (bh + max_h - 1) / max_h));
    c.w = bw / c.k, c.h = bh / c.k, c.x0 = x0, c.y0 = y0;
    if (c.w < 4 || c.h < 4) return false;
    c.y.resize(size_t(c.w) * c.h);
    const float norm = 1.f / float(c.k * c.k);
    for (int r = 0; r < c.h; ++r)
        for (int q = 0; q < c.w; ++q) {
            int s = 0;
            for (int a = 0; a < c.k; ++a) {
                const uint8_t *p = f.y + size_t(y0 + r * c.k + a) * f.y_stride + x0 + q * c.k;
                for (int b = 0; b < c.k; ++b) s += p[b];
            }
            c.y[size_t(r) * c.w + q] = float(s) * norm;
        }
    if (chroma) {
        c.cb.resize(c.y.size()), c.cr.resize(c.y.size());
        for (int r = 0; r < c.h; ++r)
            for (int q = 0; q < c.w; ++q) {
                const int fy = std::min(f.height - 1, y0 + r * c.k + c.k / 2), fx = std::min(f.width - 1, x0 + q * c.k + c.k / 2);
                const uint8_t *p = f.uv + size_t(fy / 2) * f.uv_stride + size_t(fx / 2) * 2;
                c.cb[size_t(r) * c.w + q] = p[0], c.cr[size_t(r) * c.w + q] = p[1];
            }
    }
    return true;
}

struct Blob {
    double sw = 0, sx = 0, sy = 0, sxx = 0;  // weight (luma above the threshold), its moments
    int area = 0, x0 = 1 << 30, x1 = -1, y0 = 1 << 30, y1 = -1;
    double cb = 0, cr = 0;
    int ncol = 0;
    float cx() const { return float(sx / sw); }
    float cy() const { return float(sy / sw); }
};

}  // namespace

const char *lamp_colour_name(int c) { return c == 1 ? "red" : c == 2 ? "white" : "unknown"; }

bool ru_lamps(const Nv12View &f, const RoadUserDet &u, LampPair &out, const ViewConfig &cfg) {
    out = LampPair();
    Crop c;
    if (!crop_box(f, u, cfg.max_w, cfg.max_h, true, c)) return false;
    const int n = c.w * c.h;
    int hist[256] = {};
    for (float v : c.y) ++hist[std::min(255, std::max(0, int(v + 0.5f)))];
    int bg = 0, peak = 255;
    for (int a = 0, i = 0; i < 256; ++i)
        if ((a += hist[i]) * 2 >= n) {
            bg = i;
            break;
        }
    for (int a = 0, i = 255; i >= 0; --i)
        if ((a += hist[i]) * 50 >= n) {  // 98th percentile
            peak = i;
            break;
        }
    if (bg > cfg.max_bg || peak < cfg.min_peak || peak - bg < cfg.min_contrast) return false;
    const float thr = float(bg) + 0.5f * float(peak - bg), halo = float(bg) + 0.25f * float(peak - bg);
    // connected components (8-neighbours) of the pixels above the threshold
    std::vector<int> lab(size_t(n), -1), stack;
    std::vector<Blob> blobs;
    for (int i = 0; i < n; ++i) {
        if (lab[size_t(i)] >= 0 || !(c.y[size_t(i)] > thr)) continue;
        const int id = int(blobs.size());
        blobs.emplace_back();
        Blob &b = blobs.back();
        stack.assign(1, i);
        lab[size_t(i)] = id;
        while (!stack.empty()) {
            const int p = stack.back();
            stack.pop_back();
            const int px = p % c.w, py = p / c.w;
            const double wgt = c.y[size_t(p)] - thr;
            b.sw += wgt, b.sx += wgt * px, b.sy += wgt * py, b.sxx += wgt * px * px, ++b.area;
            b.x0 = std::min(b.x0, px), b.x1 = std::max(b.x1, px), b.y0 = std::min(b.y0, py), b.y1 = std::max(b.y1, py);
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int qx = px + dx, qy = py + dy;
                    if (qx < 0 || qy < 0 || qx >= c.w || qy >= c.h) continue;
                    const int q = qy * c.w + qx;
                    if (lab[size_t(q)] < 0 && c.y[size_t(q)] > thr) lab[size_t(q)] = id, stack.push_back(q);
                }
        }
    }
    if (blobs.size() < 2) return false;
    // colour of a blob: the chroma of its halo (the core is clipped to white): the pixels above the halo level in
    // its bounding box widened by 2
    for (Blob &b : blobs) {
        for (int y = std::max(0, b.y0 - 2); y <= std::min(c.h - 1, b.y1 + 2); ++y)
            for (int x = std::max(0, b.x0 - 2); x <= std::min(c.w - 1, b.x1 + 2); ++x) {
                const size_t i = size_t(y) * c.w + x;
                if (c.y[i] > halo && c.y[i] < 235.f) b.cb += c.cb[i], b.cr += c.cr[i], ++b.ncol;
            }
        if (b.ncol) b.cb = b.cb / b.ncol - 128.0, b.cr = b.cr / b.ncol - 128.0;
    }
    // red: a clear red cast; white: next to none (headlights are a little warm: Cr up to 8 with Cb down to -10);
    // between the two (a dim tail lamp, amber, sodium light on a reflector) the colour is not known
    const auto colour_of = [](const Blob &b) {
        if (!b.ncol) return 2;  // all clipped: white
        if (b.cr >= 14 && b.cr >= -0.8 * b.cb) return 1;
        if (std::fabs(b.cr) <= 8 && b.cb >= -10 && b.cb <= 10) return 2;
        return 0;
    };
    // the pair: two blobs at one height, of like size, about the middle of the box, 0.3-0.98 of its width apart;
    // the pair with the most light in its weaker blob
    const float bw = u.x1 - u.x0, mid = 0.5f * (u.x0 + u.x1);
    int bi = -1, bj = -1;
    double best = 0;
    for (size_t i = 0; i < blobs.size(); ++i)
        for (size_t j = 0; j < blobs.size(); ++j) {
            const Blob &a = blobs[i], &b = blobs[j];
            if (i == j || !(a.cx() < b.cx())) continue;
            const float sep = c.fx(b.cx()) - c.fx(a.cx()), dy = std::fabs(c.fy(b.cy()) - c.fy(a.cy()));
            if (sep < 0.3f * bw || sep > 0.98f * bw || sep < 4.f) continue;
            if (dy > std::max(1.5f * c.k, 0.12f * sep)) continue;
            if (a.area > 10 * b.area + 2 || b.area > 10 * a.area + 2) continue;
            const bool cut = (u.clip | u.trunc) & (1 | 4);
            if (!cut && std::fabs(0.5f * (c.fx(a.cx()) + c.fx(b.cx())) - mid) > 0.25f * bw) continue;
            const int ca = colour_of(a), cb = colour_of(b);
            if (ca && cb && ca != cb) continue;  // a red and a white blob are not one vehicle's pair
            const double sc = std::min(a.sw, b.sw);
            if (sc > best) best = sc, bi = int(i), bj = int(j);
        }
    if (bi < 0) return false;
    const Blob &a = blobs[size_t(bi)], &b = blobs[size_t(bj)];
    out.ok = true;
    out.xl = c.fx(a.cx()), out.yl = c.fy(a.cy()), out.xr = c.fx(b.cx()), out.yr = c.fy(b.cy());
    out.sep = std::hypot(out.xr - out.xl, out.yr - out.yl);
    const int ca = colour_of(a), cb = colour_of(b);
    out.colour = ca == cb ? ca : (ca == 1 || cb == 1) ? 1 : 0;  // red with a dim partner: red; white needs both
    // 1-sigma of the separation. Pixel noise through the weights (3 grey levels: sigma^2 sum (x - xc)^2 / (sum w)^2)
    // is far below what the blobs' changing shapes do: measured on a night ride (residuals of ln(separation) about
    // its local trend) 0.3 px per separation of 12-20 px and about 2 % of a separation above 40 px (brake lamps and
    // indicators coming on, reflections joining a blob). Hence a floor per centroid and a share of the separation.
    const auto cen = [&](const Blob &q) {
        const double var_x = std::max(0.0, q.sxx / q.sw - (q.sx / q.sw) * (q.sx / q.sw));
        const double noise = 3.0 * std::sqrt(var_x * q.area) / std::max(1e-6, q.sw) * c.k;
        return std::max({double(cfg.cen_floor_px), 0.15 * c.k, noise});
    };
    out.sep_std = float(std::hypot(std::hypot(cen(a), cen(b)), double(cfg.sep_rel) * out.sep));
    return true;
}

float ru_symmetry(const Nv12View &f, const RoadUserDet &u, const ViewConfig &cfg) {
    if (u.x1 - u.x0 < cfg.sym_min_w || u.y1 - u.y0 < cfg.sym_min_h) return -2.f;
    constexpr int W = 32, H = 16, S = 4;
    const int x0 = std::max(0, int(std::floor(u.x0))), y0 = std::max(0, int(std::floor(u.y0)));
    const int x1 = std::min(f.width, int(std::ceil(u.x1))), y1 = std::min(f.height, int(std::ceil(u.y1)));
    if (x1 - x0 < cfg.sym_min_w || y1 - y0 < cfg.sym_min_h) return -2.f;
    float g[H][W];
    for (int r = 0; r < H; ++r) {  // area average of the box into 32 x 16 (at least one source pixel per cell)
        const int ya = y0 + r * (y1 - y0) / H, yb = std::max(ya + 1, y0 + (r + 1) * (y1 - y0) / H);
        for (int q = 0; q < W; ++q) {
            const int xa = x0 + q * (x1 - x0) / W, xb = std::max(xa + 1, x0 + (q + 1) * (x1 - x0) / W);
            int s = 0;
            for (int y = ya; y < yb; ++y) {
                const uint8_t *p = f.y + size_t(y) * f.y_stride;
                for (int x = xa; x < xb; ++x) s += p[x];
            }
            g[r][q] = float(s) / float((yb - ya) * (xb - xa));
        }
    }
    float best = -1.f;
    for (int s = -S; s <= S; ++s) {  // mirror axis at (W - 1 + s) / 2: column q pairs with W - 1 + s - q
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        int m = 0;
        for (int q = 0; q < W; ++q) {
            const int q2 = W - 1 + s - q;
            if (q2 < 0 || q2 >= W || q2 <= q) continue;  // each pair once, no column with itself
            for (int r = 0; r < H; ++r) {
                const double a = g[r][q], b = g[r][q2];
                sa += a, sb += b, saa += a * a, sbb += b * b, sab += a * b, ++m;
            }
        }
        const double va = saa - sa * sa / m, vb = sbb - sb * sb / m;
        if (m < H * 8 || va < 1e-3 || vb < 1e-3) continue;
        best = std::max(best, float((sab - sa * sb / m) / std::sqrt(va * vb)));
    }
    return best;
}

float frame_median_luma(const Nv12View &f) {
    int hist[256] = {}, n = 0;
    for (int y = 4; y < f.height; y += 8) {
        const uint8_t *p = f.y + size_t(y) * f.y_stride;
        for (int x = 4; x < f.width; x += 8) ++hist[p[x]], ++n;
    }
    for (int a = 0, i = 0; i < 256; ++i)
        if ((a += hist[i]) * 2 >= n) return float(i);
    return 255.f;
}

void ru_view(const Nv12View &f, std::vector<RoadUserDet> &users, const ViewConfig &cfg) {
    bool any = false;
    for (const RoadUserDet &u : users) any |= u.cls >= 2 && u.cls <= 5;
    const bool night = any && frame_median_luma(f) <= cfg.night_luma;
    for (RoadUserDet &u : users) {
        u.view = true;
        u.sym = -2.f;
        u.lamps = LampPair();
        if (u.cls < 2 || u.cls > 5) continue;  // vehicles: car, truck, bus, train
        u.sym = ru_symmetry(f, u, cfg);
        if (night) ru_lamps(f, u, u.lamps, cfg);
    }
}

// ------------------------------------------------------------- appearance ---

bool ru_appearance_yuv(const Nv12View &f, const RoadUserDet &u, Appearance &out) {
    out.n = 0;
    const float bw = u.x1 - u.x0, bh = u.y1 - u.y0;
    if (bw < 8.f || bh < 8.f) return false;
    const float x0 = u.x0 + 0.05f * bw, y0 = u.y0 + 0.05f * bh, cw = 0.9f * bw / 3.f, ch = 0.9f * bh / 3.f;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            const int xa = std::max(0, int(std::lround(x0 + c * cw))), xb = std::min(f.width, std::max(xa + 1, int(std::lround(x0 + (c + 1) * cw))));
            const int ya = std::max(0, int(std::lround(y0 + r * ch))), yb = std::min(f.height, std::max(ya + 1, int(std::lround(y0 + (r + 1) * ch))));
            if (xb <= xa || yb <= ya) return false;
            const int sx = std::max(1, (xb - xa) / 16), sy = std::max(1, (yb - ya) / 16);  // at most ~16 x 16 samples a cell
            long sY = 0, sU = 0, sV = 0;
            int n = 0;
            for (int y = ya; y < yb; y += sy) {
                const uint8_t *py = f.y + size_t(y) * f.y_stride, *puv = f.uv + size_t(y / 2) * f.uv_stride;
                for (int x = xa; x < xb; x += sx) sY += py[x], sU += puv[(x / 2) * 2], sV += puv[(x / 2) * 2 + 1], ++n;
            }
            const auto q = [](double v) { return int8_t(std::max(-127.0, std::min(127.0, std::round(v * 32.0)))); };
            const int k = r * 3 + c;
            out.v[k] = q((double(sY) / n - 110.0) / 50.0);
            out.v[9 + k] = q((double(sU) / n - 128.0) / 20.0);
            out.v[18 + k] = q((double(sV) / n - 128.0) / 20.0);
        }
    out.n = 27;
    return true;
}

bool ru_appearance_crop(const Nv12View &f, const RoadUserDet &u, float *rgb) {
    const float bw = u.x1 - u.x0, bh = u.y1 - u.y0;
    if (bw < 8.f || bh < 8.f) return false;
    const double x0 = std::max(0.0, double(u.x0) - 0.05 * bw), y0 = std::max(0.0, double(u.y0) - 0.05 * bh);
    const double x1 = std::min(double(f.width), double(u.x1) + 0.05 * bw), y1 = std::min(double(f.height), double(u.y1) + 0.05 * bh);
    nv12_region_to_rgb_planar(f, x0, y0, x1 - x0, y1 - y0, 32, 32, rgb, 32, size_t(32) * 32, 0, 0);
    return true;
}

std::string appearance_hex(const Appearance &a) {
    static const char *h = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < a.n; ++i) s += h[(uint8_t(a.v[i]) >> 4) & 15], s += h[uint8_t(a.v[i]) & 15];
    return s;
}

}  // namespace signs
}  // namespace roadsense
