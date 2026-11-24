// detin_test: checks of signd/detin.h that need no model (flags, the low-score list's message key, ...).
// Exit code 0 = all passed.
//   cmake --build build --target detin_test && build/detin_test
//   build/detin_test --flags W H < boxes   evaluation helper: lines "image x0 y0 x1 y1 [clip-bits]" (one box per
//                                          line, boxes of one image on consecutive lines) -> the same lines with
//                                          " trunc-bits occ occ_by occ_edges-bits" appended (signd's own code on
//                                          boxes from elsewhere, e.g. a data set's labels)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "detin.h"
#include "lights.h"

using namespace roadsense::signs;

namespace {
int failures = 0;
#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                  \
        }                                                                \
    } while (0)

RoadUserDet box(int cls, float x0, float y0, float x1, float y1, float score = 0.8f) {
    RoadUserDet d;
    d.x0 = x0, d.y0 = y0, d.x1 = x1, d.y1 = y1, d.score = score, d.cls = cls;
    for (float &p : d.p) p = 0.f;
    d.p[cls] = score;
    return d;
}

// A dark 640 x 360 NV12 frame with Gaussian blobs: (x, y, sigma, peak luma above the background, Cb, Cr).
struct Blob6 {
    double x, y, s, a;
    int cb, cr;
};
struct Frame {
    int w = 640, h = 360;
    std::vector<uint8_t> d;
    Frame(int bg, const std::vector<Blob6> &blobs) : d(size_t(640) * 360 * 3 / 2, 128) {
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                double v = bg + 3.0 * std::sin(0.7 * x) * std::cos(0.5 * y);
                for (const Blob6 &b : blobs) {
                    const double g = b.a * std::exp(-((x - b.x) * (x - b.x) + (y - b.y) * (y - b.y)) / (2 * b.s * b.s));
                    v += g;
                    if (g > 10) {
                        uint8_t *uv = d.data() + size_t(w) * h + size_t(y / 2) * w + size_t(x / 2) * 2;
                        uv[0] = uint8_t(b.cb), uv[1] = uint8_t(b.cr);
                    }
                }
                d[size_t(y) * w + x] = uint8_t(std::min(255.0, std::max(0.0, std::round(v))));
            }
    }
    Nv12View view() const {
        Nv12View v;
        v.y = d.data(), v.uv = d.data() + size_t(w) * h, v.width = w, v.height = h, v.y_stride = v.uv_stride = w;
        return v;
    }
};

int flags_tool(int fw, int fh) {
    char line[512], img[256], last[256] = "";
    std::vector<RoadUserDet> v;
    std::vector<std::string> lines;
    auto flush = [&]() {
        ru_flags(v, nullptr, fw, fh);
        for (size_t i = 0; i < v.size(); ++i)
            printf("%s %d %.3f %d %d\n", lines[i].c_str(), v[i].trunc, v[i].occ, v[i].occ_by, v[i].occ_edges);
        v.clear(), lines.clear();
    };
    while (fgets(line, sizeof(line), stdin)) {
        float x0, y0, x1, y1;
        int clip = 0;
        const int n = sscanf(line, "%255s %f %f %f %f %d", img, &x0, &y0, &x1, &y1, &clip);
        if (n < 5) continue;
        if (strcmp(img, last) != 0) flush(), strcpy(last, img);
        RoadUserDet d = box(2, x0, y0, x1, y1);
        d.clip = clip;
        v.push_back(d);
        line[strcspn(line, "\r\n")] = 0;
        lines.push_back(line);
    }
    flush();
    return 0;
}
}  // namespace

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "--flags")) return flags_tool(atoi(argv[2]), atoi(argv[3]));
    {   // a far car half behind a near one: the far one is covered, the near one is not
        std::vector<RoadUserDet> v = {box(2, 100, 100, 200, 180), box(2, 150, 90, 230, 150)};
        ru_flags(v, nullptr, 1280, 720);
        CHECK(ru_nearer(v[0], v[1]) == 1 && ru_nearer(v[1], v[0]) == -1);
        CHECK(v[0].flags && v[0].occ == 0 && v[0].occ_by == -1 && v[0].occ_edges == 0);
        // intersection x 150..200 of 150..230 (62 %), y 100..150 of 90..150 (83 %): about half the box, its left
        // and bottom edges
        CHECK(v[1].occ > 0.45f && v[1].occ < 0.6f && v[1].occ_by == 0);
        CHECK(v[1].occ_edges == (1 | 8));
        CHECK(v[0].trunc == 0 && v[1].trunc == 0);
    }
    {   // bottoms within the tie: not ordered, each counts the other
        std::vector<RoadUserDet> v = {box(2, 100, 100, 200, 180), box(2, 180, 100, 280, 181)};
        ru_flags(v, nullptr, 1280, 720);
        CHECK(ru_nearer(v[0], v[1]) == 0);
        CHECK(v[0].occ > 0.15f && v[0].occ_by == 1 && v[1].occ > 0.15f && v[1].occ_by == 0);
        CHECK(v[0].occ_edges == 4 && v[1].occ_edges == 1);
    }
    {   // a bottom cut by the view is lower than it shows: that box is the nearer one
        std::vector<RoadUserDet> v = {box(2, 100, 300, 400, 446), box(2, 350, 380, 450, 440)};
        v[0].clip = 8;
        ru_flags(v, nullptr, 1280, 720);
        CHECK(ru_nearer(v[0], v[1]) == 1);
        CHECK(v[0].occ == 0 && v[1].occ > 0.4f && v[1].occ_by == 0);
    }
    {   // frame edges; a person in front of a bus covers a little of it and none of its edges
        std::vector<RoadUserDet> v = {box(4, 0, 50, 300, 250), box(0, 120, 150, 150, 260), box(2, 1200, 600, 1280, 720)};
        ru_flags(v, nullptr, 1280, 720);
        CHECK(v[0].trunc == 1 && v[1].trunc == 0 && v[2].trunc == (4 | 8));
        CHECK(v[0].occ > 0.02f && v[0].occ < 0.1f && v[0].occ_by == 1 && v[0].occ_edges == 0);
        CHECK(v[1].occ == 0 && v[2].occ == 0);
    }
    {   // low boxes are flagged against the listed ones and are never occluders
        std::vector<RoadUserDet> v = {box(2, 100, 100, 200, 180)}, low = {box(2, 150, 90, 230, 150, 0.2f), box(2, 90, 120, 210, 200, 0.2f)};
        ru_flags(v, &low, 1280, 720);
        CHECK(v[0].occ == 0);
        CHECK(low[0].flags && low[0].occ > 0.45f && low[0].occ_by == 0 && low[1].occ == 0);
    }
    {   // message keys: nothing without the flags; trunc always with them, the occ keys only when covered
        LightsConfig cfg;
        std::vector<LightTrackReport> none;
        std::vector<RoadUserDet> v = {box(2, 100, 100, 200, 180), box(2, 150, 90, 230, 150)}, low = {box(3, 500, 100, 560, 140, 0.2f)};
        const std::string off = lights_json(1, 2, 3, 1.0, 1280, 720, cfg, none, nullptr, nullptr, 0, &v, 1);
        CHECK(off.find("trunc") == std::string::npos && off.find("occ") == std::string::npos && off.find("road_users_low") == std::string::npos);
        ru_flags(v, &low, 1280, 720);
        const std::string on = lights_json(1, 2, 3, 1.0, 1280, 720, cfg, none, nullptr, nullptr, 0, &v, 1, &low);
        CHECK(on.find("\"p\":[0.00,0.00,0.80,0.00,0.00,0.00,0.00,0.00],\"trunc\":\"\"},{") != std::string::npos);
        CHECK(on.find(",\"trunc\":\"\",\"occ\":0.51,\"occ_by\":0,\"occ_edges\":\"lb\"}]") != std::string::npos);
        CHECK(on.find(",\"road_users_low\":[{\"cls\":\"truck\",\"score\":0.200,") != std::string::npos);
        // shed road users: no low list either
        const std::string shed = lights_json(1, 2, 3, 1.0, 1280, 720, cfg, none, nullptr, nullptr, 0, &v, 2, &low);
        CHECK(shed.find("road_users_low") == std::string::npos);
    }
    {   // lamp pair: two red blobs at one height on a dark body -> the pair, its centroids, red
        const Frame f(25, {{300.3, 180.2, 2.2, 600, 112, 170}, {326.8, 180.6, 2.2, 600, 112, 170}});
        RoadUserDet u = box(2, 290, 165, 338, 198);
        LampPair lp;
        CHECK(ru_lamps(f.view(), u, lp));
        CHECK(lp.ok && lp.colour == 1);
        CHECK(std::fabs(lp.xl - 300.3) < 0.3 && std::fabs(lp.xr - 326.8) < 0.3 && std::fabs(lp.yl - 180.2) < 0.3);
        CHECK(std::fabs(lp.sep - 26.5) < 0.4 && lp.sep_std > 0.2 && lp.sep_std < 1.0);
        std::vector<RoadUserDet> v = {u, box(0, 100, 100, 120, 160)};
        ru_view(f.view(), v);
        CHECK(v[0].view && v[0].lamps.ok && v[0].sym > 0.6);    // symmetric: two equal blobs about the middle
        CHECK(v[1].view && !v[1].lamps.ok && v[1].sym < -1.f);  // a person: no vehicle cues
        LightsConfig cfg;
        std::vector<LightTrackReport> none;
        const std::string m = lights_json(1, 2, 3, 1.0, 640, 360, cfg, none, nullptr, nullptr, 0, &v, 1);
        CHECK(m.find(",\"sym\":") != std::string::npos && m.find(",\"lamps\":{\"c\":\"red\",\"l\":[300.") != std::string::npos);
        CHECK(m.find("\"sep\":26.") != std::string::npos);
    }
    {   // white blobs -> white; one blob -> no pair; blobs at different heights -> no pair
        const Frame f(30, {{200, 100, 3, 700, 128, 128}, {240, 100.5, 3, 700, 128, 128}, {400, 100, 3, 700, 128, 128},
                           {500, 200, 2.5, 700, 128, 128}, {530, 214, 2.5, 700, 128, 128}});
        LampPair lp;
        CHECK(ru_lamps(f.view(), box(2, 185, 80, 255, 125), lp) && lp.colour == 2 && std::fabs(lp.sep - 40.0) < 0.5);
        CHECK(!ru_lamps(f.view(), box(2, 380, 80, 440, 125), lp) && !lp.ok);
        CHECK(!ru_lamps(f.view(), box(2, 488, 185, 542, 230), lp));
    }
    {   // a lit scene: the frame is not dark, no lamp search; a red and a white blob are not a pair
        const Frame day(140, {{300, 180, 2.2, 600, 112, 170}, {326, 180, 2.2, 600, 112, 170}});
        std::vector<RoadUserDet> v = {box(2, 290, 165, 338, 198)};
        ru_view(day.view(), v);
        CHECK(frame_median_luma(day.view()) > 120 && !v[0].lamps.ok);
        const Frame mix(25, {{300, 180, 2.2, 600, 112, 170}, {326, 180, 2.2, 600, 128, 128}});
        LampPair lp;
        CHECK(!ru_lamps(mix.view(), box(2, 290, 165, 338, 198), lp));
    }
    {   // symmetry: a box symmetric about its middle scores high, a one-sided one low, a tiny one is not measured
        const Frame f(40, {{300, 150, 6, 150, 128, 128}, {340, 150, 6, 150, 128, 128}, {320, 165, 9, 80, 128, 128},
                           {470, 150, 7, 180, 128, 128}});
        CHECK(ru_symmetry(f.view(), box(2, 280, 130, 360, 185)) > 0.9f);
        CHECK(ru_symmetry(f.view(), box(2, 455, 130, 535, 185)) < 0.3f);
        CHECK(ru_symmetry(f.view(), box(2, 300, 140, 310, 146)) < -1.f);
    }
    {   // appearance vector: 27 numbers of a uniform patch; alike for a shifted box of the same thing, far from another
        Frame f(60, {});
        for (int y = 100; y < 160; ++y)
            for (int x = 200; x < 280; ++x) {
                f.d[size_t(y) * f.w + x] = 160;
                uint8_t *uv = f.d.data() + size_t(f.w) * f.h + size_t(y / 2) * f.w + size_t(x / 2) * 2;
                uv[0] = 108, uv[1] = 148;
            }
        Appearance a, b, c;
        CHECK(ru_appearance_yuv(f.view(), box(2, 200, 100, 280, 160), a) && a.n == 27);
        CHECK(a.v[4] == 32 && a.v[13] == -32 && a.v[22] == 32);   // (160 - 110) / 50, (108 - 128) / 20, (148 - 128) / 20, x 32
        CHECK(ru_appearance_yuv(f.view(), box(2, 203, 102, 279, 158), b));
        CHECK(ru_appearance_yuv(f.view(), box(2, 400, 100, 480, 160), c));
        double dab = 0, dac = 0;
        for (int i = 0; i < 27; ++i) dab += (a.v[i] - b.v[i]) * (a.v[i] - b.v[i]), dac += (a.v[i] - c.v[i]) * (a.v[i] - c.v[i]);
        CHECK(std::sqrt(dab) < 6 && std::sqrt(dac) > 150);
        CHECK(appearance_hex(a).size() == 54 && appearance_hex(a).substr(8, 2) == "20" && appearance_hex(a).substr(26, 2) == "e0");
        CHECK(!ru_appearance_yuv(f.view(), box(2, 200, 100, 206, 160), c) && c.n == 0);
        float rgb[3 * 32 * 32];
        CHECK(ru_appearance_crop(f.view(), box(2, 200, 100, 280, 160), rgb) && rgb[16 * 32 + 16] > 0.5f);
        std::vector<RoadUserDet> v = {box(2, 200, 100, 280, 160)};
        v[0].emb = a;
        LightsConfig cfg;
        std::vector<LightTrackReport> none;
        const std::string off = lights_json(1, 2, 3, 1.0, 640, 360, cfg, none, nullptr, nullptr, 0, &v, 1);
        const std::string on = lights_json(1, 2, 3, 1.0, 640, 360, cfg, none, nullptr, nullptr, 0, &v, 1, nullptr, "yuv3x3");
        CHECK(off.find("emb_kind") == std::string::npos && off.find(",\"emb\":\"") != std::string::npos);
        CHECK(on.find("}],\"emb_kind\":\"yuv3x3\"") != std::string::npos);
    }
    if (failures) fprintf(stderr, "detin_test: %d failure(s)\n", failures);
    else fprintf(stderr, "detin_test: all passed\n");
    return failures ? 1 : 0;
}
