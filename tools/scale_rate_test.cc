// scale_rate_test: synthetic checks of the road-user scale rate (signd/scale.h), no model needed.
//
// A textured rectangle (the "road user") grows at a known log rate and moves across a static textured
// background; frames are rendered by 4x4 supersampling (pixel area integration) with Gaussian sensor noise, the
// boxes the estimator gets carry edge jitter like the detector's. Five passes a second for 1.2 s, as the light
// thread runs; the estimator sees every pass. Per case, over many random trials: bias of scale_rate, actual
// scatter, mean stated std, pull (error / stated std) scatter, coverage. A uniform object and an object cut by
// the frame edge must give no value. Exit code 0 = all checks passed.
//   cmake --build build --target scale_rate_test && build/scale_rate_test [-v]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "lights.h"
#include "scale.h"

using namespace roadsense::signs;

namespace {
int failures = 0;
#define CHECK(cond)                                                           \
    do {                                                                      \
        if (!(cond)) {                                                        \
            fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                       \
        }                                                                     \
    } while (0)

constexpr int W = 1280, H = 720, TX = 128;

struct Tex {
    std::vector<float> v;
    int n;
    float at(double x, double y) const {  // x, y in texels, clamped
        x = std::min(std::max(x, 0.0), n - 1.001), y = std::min(std::max(y, 0.0), n - 1.001);
        const int x0 = int(x), y0 = int(y);
        const float fx = float(x - x0), fy = float(y - y0);
        const float *r0 = &v[size_t(y0) * n + x0], *r1 = r0 + n;
        const float a = r0[0] + fx * (r0[1] - r0[0]), b = r1[0] + fx * (r1[1] - r1[0]);
        return a + fy * (b - a);
    }
};

Tex make_tex(std::mt19937 &rng, int n, float base, float contrast, int blobs, float smin, float smax) {
    Tex t;
    t.n = n;
    t.v.assign(size_t(n) * n, base);
    std::uniform_real_distribution<float> U(0, 1);
    for (int b = 0; b < blobs; ++b) {
        const float cx = U(rng) * n, cy = U(rng) * n, s = smin + U(rng) * (smax - smin), a = (U(rng) * 2 - 1) * contrast;
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                const float d2 = ((x - cx) * (x - cx) + (y - cy) * (y - cy)) / (2 * s * s);
                if (d2 < 9) t.v[size_t(y) * n + x] += a * std::exp(-d2);
            }
    }
    return t;
}

struct Case {
    const char *name;
    double h0, aspect;    // object height at t = 0 (px), width / height
    double rate;          // d ln size / dt (1/s)
    double vx, vy;        // image motion (px/s)
    float contrast, noise;
    bool uniform = false;  // textureless object
    double x_start = 600;  // object centre x at t = 0
    double jitter = 2.0;   // box edge jitter (px, 1 sigma)
    // A vehicle in the dark: a dark body (luma about 25, faint texture), two lamps at +- 0.32 of the width (cores
    // clipped, a halo whose width changes from frame to frame: blooming with the exposure), the exposure itself
    // changing by +- 20 % per frame, horizontal motion blur of 0-3 px, red chroma on the lamps, noise `noise`.
    bool night = false;
};
enum Mode { kPatch = 0, kMulti3, kLamps, kLampsOnly, kLampsMulti };
const char *mode_name(int m) {
    static const char *n[] = {"patch", "patch x3", "patch+lamps", "lamps only", "patch x3+lamps"};
    return n[m];
}

struct Stats {
    int n = 0, measured = 0;
    double se = 0, se2 = 0, sstd = 0, sp2 = 0, sdt = 0;
    int out3 = 0;  // |error| > 3 stated sigma
    double ms = 0;  // update() time of the passes with a measurement attempt
    int ms_n = 0;
    std::vector<double> err;
};

void render(const Case &c, const Tex &obj, const Tex &bg, double t, std::mt19937 &rng, std::vector<uint8_t> &img,
            double box[4]) {
    const double h = c.h0 * std::exp(c.rate * t), w = h * c.aspect;
    const double cx = c.x_start + c.vx * t, cy = 380 + c.vy * t;
    box[0] = cx - w / 2, box[1] = cy - h / 2, box[2] = cx + w / 2, box[3] = cy + h / 2;
    std::normal_distribution<float> N(0, c.noise);
    if (img.size() != size_t(W) * H * 3 / 2) {  // static background, rendered once per trial
        img.resize(size_t(W) * H * 3 / 2);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) img[size_t(y) * W + x] = uint8_t(std::lround(bg.at(x / 10.0, y / 10.0)));
    }
    // only the neighbourhood of the object is re-rendered (with noise); the estimator reads nothing else
    const int x0 = std::max(0, int(box[0]) - 40), x1 = std::min(W, int(box[2]) + 40);
    const int y0 = std::max(0, int(box[1]) - 40), y1 = std::min(H, int(box[3]) + 40);
    for (int y = std::max(0, y0 - 30); y < std::min(H, y1 + 30); ++y)
        for (int x = std::max(0, x0 - 30); x < std::min(W, x1 + 30); ++x) {
            float v;
            if (x >= x0 && x < x1 && y >= y0 && y < y1) {
                v = 0;
                for (int a = 0; a < 4; ++a)
                    for (int b = 0; b < 4; ++b) {
                        const double px = x + (b + 0.5) / 4 - 0.5, py = y + (a + 0.5) / 4 - 0.5;  // pixel centre at x
                        const double ox = (px - box[0]) / w, oy = (py - box[1]) / h;
                        if (ox >= 0 && ox < 1 && oy >= 0 && oy < 1) v += c.uniform ? 80.f : obj.at(ox * (obj.n - 1), oy * (obj.n - 1));
                        else v += bg.at(px / 10.0, py / 10.0);
                    }
                v /= 16;
            } else {
                v = bg.at(x / 10.0, y / 10.0);
            }
            v += N(rng);
            img[size_t(y) * W + x] = uint8_t(std::min(255.f, std::max(0.f, std::round(v))));
        }
    std::memset(img.data() + size_t(W) * H, 128, size_t(W) * H / 2);
}

// The night scene (Case::night). Lamp: A exp(-d^2 / 2 s^2), s^2 = (0.05 w)^2 + bloom^2, A = 700 (clipped).
void render_night(const Case &c, const Tex &obj, const Tex &bg, double t, std::mt19937 &rng, std::vector<uint8_t> &img,
                  double box[4]) {
    const double h = c.h0 * std::exp(c.rate * t), w = h * c.aspect;
    const double cx = c.x_start + c.vx * t, cy = 380 + c.vy * t;
    box[0] = cx - w / 2, box[1] = cy - h / 2, box[2] = cx + w / 2, box[3] = cy + h / 2;
    std::normal_distribution<float> N(0, c.noise);
    std::uniform_real_distribution<double> U(0, 1);
    const double gain = 0.8 + 0.4 * U(rng), bloom = 0.6 + 1.2 * U(rng), blur = 3.0 * U(rng);
    if (img.size() != size_t(W) * H * 3 / 2) {
        img.resize(size_t(W) * H * 3 / 2);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) img[size_t(y) * W + x] = uint8_t(std::lround(0.4f * bg.at(x / 10.0, y / 10.0)));
    }
    std::memset(img.data() + size_t(W) * H, 128, size_t(W) * H / 2);
    const double ls2 = (0.05 * w) * (0.05 * w) + bloom * bloom, ly = box[1] + 0.45 * h;
    const double lx[2] = {cx - 0.32 * w, cx + 0.32 * w};
    const int x0 = std::max(0, int(box[0]) - 40), x1 = std::min(W, int(box[2]) + 40);
    const int y0 = std::max(0, int(box[1]) - 40), y1 = std::min(H, int(box[3]) + 40);
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) {
            double v = 0, lamp = 0;
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b)
                    for (int m = 0; m < 3; ++m) {  // motion blur: three samples along x
                        const double px = x + (b + 0.5) / 4 - 0.5 + (m - 1) * blur / 3, py = y + (a + 0.5) / 4 - 0.5;
                        const double ox = (px - box[0]) / w, oy = (py - box[1]) / h;
                        double q = (ox >= 0 && ox < 1 && oy >= 0 && oy < 1) ? 25.0 + 0.15 * (obj.at(ox * (obj.n - 1), oy * (obj.n - 1)) - 90.0)
                                                                         : 0.4 * bg.at(px / 10.0, py / 10.0);
                        double l = 0;
                        for (double lxk : lx) l += 700.0 * std::exp(-((px - lxk) * (px - lxk) + (py - ly) * (py - ly)) / (2 * ls2));
                        v += q + l, lamp += l;
                    }
            v = v / 48 * gain + N(rng), lamp = lamp / 48 * gain;
            img[size_t(y) * W + x] = uint8_t(std::min(255.0, std::max(0.0, std::round(v))));
            if (lamp > 8) {  // red halo: Cr up, Cb down, strongest where the luma is not clipped
                uint8_t *uv = img.data() + size_t(W) * H + size_t(y / 2) * W + size_t(x / 2) * 2;
                const double k = std::min(1.0, lamp / 60.0);
                uv[0] = uint8_t(128 - 18 * k), uv[1] = uint8_t(128 + 45 * k);
            }
        }
}

Stats run_case(const Case &c, int trials, unsigned seed, bool verbose, int mode = kPatch) {
    Stats st;
    std::mt19937 rng(seed);
    std::normal_distribution<double> J(0, 1);
    std::vector<uint8_t> img;
    for (int tr = 0; tr < trials; ++tr) {
        const Tex obj = make_tex(rng, TX, 90, c.contrast, 40, 3, 14);
        const Tex bg = make_tex(rng, TX + 32, 40, 15, 60, 2, 8);
        ScaleConfig cfg;
        cfg.baselines = (mode == kMulti3 || mode == kLampsMulti) ? 3 : 1;
        cfg.lamps = mode >= kLamps;
        cfg.patch = mode != kLampsOnly;
        ScaleRateEstimator est(cfg);
        img.clear();
        for (int k = 0; k <= 6; ++k) {
            const double t = 0.2 * k;
            double box[4];
            if (c.night) render_night(c, obj, bg, t, rng, img, box);
            else render(c, obj, bg, t, rng, img, box);
            Nv12View v;
            v.y = img.data(), v.uv = img.data() + size_t(W) * H, v.width = W, v.height = H, v.y_stride = v.uv_stride = W;
            RoadUserDet d;
            d.x0 = float(std::max(0.0, box[0] + c.jitter * J(rng))), d.y0 = float(std::max(0.0, box[1] + c.jitter * J(rng)));
            d.x1 = float(std::min(double(W), box[2] + c.jitter * J(rng))), d.y1 = float(std::min(double(H), box[3] + c.jitter * J(rng)));
            d.score = 0.8f, d.cls = 2;
            for (float &p : d.p) p = 0;
            std::vector<RoadUserDet> users = {d};
            const auto c0 = std::chrono::steady_clock::now();
            if (cfg.lamps) ru_lamps(v, users[0], users[0].lamps);
            est.update(v, int64_t(t * 1e9), users);
            if (k >= 1) st.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count(), ++st.ms_n;
            if (k < 2) continue;  // baselines from 0.4 s on
            ++st.n;
            if (!users[0].scale_ok) continue;
            ++st.measured;
            const double e = users[0].scale_rate - c.rate;
            st.se += e, st.se2 += e * e, st.sstd += users[0].scale_rate_std, st.sp2 += (e / users[0].scale_rate_std) * (e / users[0].scale_rate_std);
            st.sdt += users[0].scale_dt_s;
            if (std::fabs(e) > 3 * users[0].scale_rate_std) ++st.out3;
            st.err.push_back(e);
            if (verbose)
                fprintf(stderr, "  %s trial %d t %.1f: rate %.4f std %.4f dt %.2f err %+.4f\n", c.name, tr, t, users[0].scale_rate,
                        users[0].scale_rate_std, users[0].scale_dt_s, e);
        }
    }
    return st;
}
}  // namespace

int main(int argc, char **argv) {
    const bool verbose = argc > 1 && std::strcmp(argv[1], "-v") == 0;
    std::vector<Case> cases = {
        {"far 16 px, still", 16, 1.25, 0.0, 0, 0, 40, 3},
        {"far 16 px, closing", 16, 1.25, 0.15, 20, 0, 40, 3},
        {"far 24 px, low contrast", 24, 1.25, 0.10, -15, 5, 15, 3},
        {"mid 60 px, closing", 60, 1.2, 0.30, 40, 10, 40, 3},
        {"mid 60 px, opening", 60, 1.2, -0.20, -30, 0, 40, 3},
        {"near 180 px, closing", 180, 1.1, 0.40, 10, 20, 40, 3, false, 640, 4.0},
    };
    printf("%-26s %5s %8s %8s %9s %9s %8s %6s %7s %7s\n", "case", "n", "coverage", "bias", "scatter", "stated", "pull", ">3sig", "dt", "ms/box");
    unsigned seed = 1;
    for (const Case &c : cases) {
        const Stats s = run_case(c, 60, seed++, verbose);
        const double m = s.measured ? s.se / s.measured : 0;
        const double sd = s.measured > 1 ? std::sqrt(std::max(0.0, s.se2 / s.measured - m * m)) : 0;
        const double stated = s.measured ? s.sstd / s.measured : 0, pull = s.measured ? std::sqrt(s.sp2 / s.measured) : 0;
        printf("%-26s %5d %7.0f%% %+8.4f %9.4f %9.4f %8.2f %5.1f%% %7.2f %7.3f\n", c.name, s.n, 100.0 * s.measured / std::max(1, s.n),
               m, sd, stated, pull, 100.0 * s.out3 / std::max(1, s.measured), s.measured ? s.sdt / s.measured : 0,
               s.ms / std::max(1, s.ms_n));
        CHECK(s.measured >= s.n / 2);                         // a textured object gets a value most of the time
        CHECK(std::fabs(m) < 0.01 + 3 * sd / std::sqrt(std::max(1, s.measured)));  // no bias beyond the noise
        CHECK(pull > 0.5 && pull < 1.6);                      // stated std matches the scatter within a factor
        CHECK(s.out3 <= s.measured / 50 + 1);                 // no heavy tail of confident wrong values
    }
    {   // more baselines, the lamp pair: by day (several baselines), in the dark (all modes)
        const std::vector<Case> day = {cases[1], cases[2], cases[3]};
        const std::vector<Case> night = {
            {"night far 16 px, closing", 16, 1.25, 0.15, 20, 0, 40, 4, false, 600, 2.0, true},
            {"night far 16 px, still", 16, 1.25, 0.0, 0, 0, 40, 4, false, 600, 2.0, true},
            {"night mid 32 px, closing", 32, 1.25, 0.25, 30, 5, 40, 4, false, 600, 2.5, true},
            {"night mid 32 px, opening", 32, 1.25, -0.20, -20, 0, 40, 4, false, 600, 2.5, true},
            {"night near 90 px, closing", 90, 1.2, 0.35, 10, 10, 40, 4, false, 640, 3.0, true},
        };
        const auto row = [&](const Case &c, int mode, unsigned sd, bool check) {
            const Stats s = run_case(c, 40, sd, false, mode);
            const double m = s.measured ? s.se / s.measured : 0;
            const double sdv = s.measured > 1 ? std::sqrt(std::max(0.0, s.se2 / s.measured - m * m)) : 0;
            const double stated = s.measured ? s.sstd / s.measured : 0, pull = s.measured ? std::sqrt(s.sp2 / s.measured) : 0;
            printf("%-26s %-15s %5d %7.0f%% %+8.4f %9.4f %9.4f %8.2f %5.1f%% %7.2f %7.3f\n", c.name, mode_name(mode), s.n,
                   100.0 * s.measured / std::max(1, s.n), m, sdv, stated, pull, 100.0 * s.out3 / std::max(1, s.measured),
                   s.measured ? s.sdt / s.measured : 0, s.ms / std::max(1, s.ms_n));
            if (check) {
                CHECK(std::fabs(m) < 0.01 + 3 * sdv / std::sqrt(std::max(1, s.measured)));
                // the lamp pair's stated std is set by what real blobs do (signd/detin.cc), which these clean
                // Gaussians do not: it may be far above the scatter here, never below it
                CHECK(pull < 1.6 && (mode == kLampsOnly || pull > 0.5));
                CHECK(s.out3 <= s.measured / 25 + 1);
            }
            return s;
        };
        printf("%-26s %-15s %5s %8s %8s %9s %9s %8s %6s %7s %7s\n", "case", "mode", "n", "coverage", "bias", "scatter", "stated", "pull", ">3sig", "dt", "ms/box");
        unsigned sd = 200;
        for (const Case &c : day) {
            const Stats a = row(c, kPatch, sd, false);
            const Stats b = row(c, kMulti3, sd, true);
            CHECK(b.measured >= a.measured);  // more baselines never cost coverage
            ++sd;
        }
        for (const Case &c : night) {
            const Stats a = row(c, kPatch, sd, false);
            row(c, kLampsOnly, sd, true);
            const Stats b = row(c, kLamps, sd, false);
            CHECK(b.measured >= a.measured);  // the lamp pair never costs coverage
            ++sd;
        }
    }
    {   // low-score boxes: they continue an object's chain and are measured; alone they start nothing
        std::mt19937 rng(77);
        const Case c = {"low list", 60, 1.2, 0.30, 40, 10, 40, 3};
        const Tex obj = make_tex(rng, TX, 90, c.contrast, 40, 3, 14), bg = make_tex(rng, TX + 32, 40, 15, 60, 2, 8);
        ScaleConfig cfg;
        ScaleRateEstimator with(cfg), without(cfg);
        std::vector<uint8_t> img;
        bool low_measured = false, after_with = false, after_without = false;
        double dt_with = 0, dt_without = 0;
        for (int k = 0; k <= 4; ++k) {
            const double t = 0.2 * k;
            double b[4];
            render(c, obj, bg, t, rng, img, b);
            Nv12View v;
            v.y = img.data(), v.uv = img.data() + size_t(W) * H, v.width = W, v.height = H, v.y_stride = v.uv_stride = W;
            RoadUserDet d;
            d.x0 = float(b[0]), d.y0 = float(b[1]), d.x1 = float(b[2]), d.y1 = float(b[3]), d.score = 0.8f, d.cls = 2;
            for (float &p : d.p) p = 0;
            // passes 1-3: the object only in the low list (for `with`), in no list (for `without`)
            std::vector<RoadUserDet> hi, low, none;
            if (k == 0 || k == 4) hi = {d};
            else low = {d};
            std::vector<RoadUserDet> hi2 = hi;
            with.update(v, int64_t(t * 1e9), hi, &low);
            without.update(v, int64_t(t * 1e9), hi2, &none);
            if (k == 3) low_measured = low[0].scale_ok && std::fabs(low[0].scale_rate - c.rate) < 0.03;
            if (k == 4) {
                after_with = hi[0].scale_ok, dt_with = hi[0].scale_dt_s;
                after_without = hi2[0].scale_ok, dt_without = hi2[0].scale_dt_s;
            }
        }
        CHECK(low_measured);                       // a low box on a known object is measured
        CHECK(after_with && dt_with > 0.75);       // the chain went on through the low boxes: baseline 0.8 s
        CHECK(with.chains() == 1 && without.chains() == 1);
        (void)after_without, (void)dt_without;     // (without the low list the chain is 0.8 s old too: within the gap)
        // a low box where no object is known starts no chain
        ScaleRateEstimator fresh(cfg);
        RoadUserDet d;
        d.x0 = 100, d.y0 = 100, d.x1 = 160, d.y1 = 150, d.score = 0.2f, d.cls = 2;
        for (float &p : d.p) p = 0;
        std::vector<RoadUserDet> hi, low = {d};
        Nv12View v;
        v.y = img.data(), v.uv = img.data() + size_t(W) * H, v.width = W, v.height = H, v.y_stride = v.uv_stride = W;
        fresh.update(v, 0, hi, &low);
        CHECK(fresh.chains() == 0 && !low[0].scale_ok);
    }
    {   // a textureless object: no value
        const Case c = {"uniform 60 px", 60, 1.2, 0.3, 20, 0, 0, 3, true};
        const Stats s = run_case(c, 20, 99, verbose);
        printf("%-26s %5d %7.0f%%\n", c.name, s.n, 100.0 * s.measured / std::max(1, s.n));
        CHECK(s.measured == 0);
    }
    {   // cut by the left frame edge: no value
        const Case c = {"clipped at frame edge", 80, 1.2, 0.3, 0, 0, 40, 3, false, 20};
        const Stats s = run_case(c, 20, 98, verbose);
        printf("%-26s %5d %7.0f%%\n", c.name, s.n, 100.0 * s.measured / std::max(1, s.n));
        CHECK(s.measured == 0);
    }
    {   // tiny (5 px inner region, under min_side_px even with the box jitter): no value
        const Case c = {"tiny 6 px", 6, 1.2, 0.2, 0, 0, 40, 3};
        const Stats s = run_case(c, 20, 97, verbose);
        printf("%-26s %5d %7.0f%%\n", c.name, s.n, 100.0 * s.measured / std::max(1, s.n));
        CHECK(s.measured == 0);
    }
    {   // the JSON: fields only when measured
        RoadUserDet d;
        d.x0 = 1, d.y0 = 2, d.x1 = 30, d.y1 = 40, d.score = 0.9f, d.cls = 2;
        for (float &p : d.p) p = 0;
        std::vector<RoadUserDet> v = {d};
        LightsConfig cfg;
        std::vector<LightTrackReport> none;
        std::string s = lights_json(1, 2, 3, 4, 1280, 720, cfg, none, nullptr, nullptr, 0, &v, 1);
        CHECK(s.find("scale_rate") == std::string::npos);
        v[0].scale_ok = true, v[0].scale_rate = 0.1234f, v[0].scale_rate_std = 0.0123f, v[0].scale_dt_s = 0.8f;
        s = lights_json(1, 2, 3, 4, 1280, 720, cfg, none, nullptr, nullptr, 0, &v, 1);
        CHECK(s.find("\"p\":[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],\"scale_rate\":0.1234,\"scale_rate_std\":0.0123,\"scale_dt_s\":0.800}") !=
              std::string::npos);
    }
    if (failures) fprintf(stderr, "scale_rate_test: %d check(s) FAILED\n", failures);
    else printf("scale_rate_test: all checks passed\n");
    return failures ? 1 : 0;
}
