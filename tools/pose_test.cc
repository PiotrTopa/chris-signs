// pose_test FIXTURE.onnx: checks of signd/pose.h with a stand-in model (tools/pose/make_fixture.py: the mean of
// each input plane through a fixed linear map; no trained weights). What is checked: the crop (colour, black
// outside the picture), the nine box numbers, the reading of the outputs into frame pixels, the JSON, and that the
// model is loaded and run through the same runner as the other heads. Exit code 0 = all passed.
//   build/pose_test tools/pose/pose_fixture.onnx
//   build/pose_test --bench MODEL.onnx [N]     time per box of a real head (one thread): crop + run + decode
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

#include "lights.h"
#include "pose.h"

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
bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

struct Frame {
    int w, h;
    std::vector<uint8_t> d;
    Frame(int w_, int h_, int y, int cb, int cr) : w(w_), h(h_), d(size_t(w_) * h_ * 3 / 2) {
        std::fill(d.begin(), d.begin() + long(size_t(w) * h), uint8_t(y));
        for (size_t i = size_t(w) * h; i < d.size(); i += 2) d[i] = uint8_t(cb), d[i + 1] = uint8_t(cr);
    }
    Nv12View view() const {
        Nv12View v;
        v.y = d.data(), v.uv = d.data() + size_t(w) * h, v.width = w, v.height = h, v.y_stride = v.uv_stride = w;
        return v;
    }
};

RoadUserDet box(float x0, float y0, float x1, float y1) {
    RoadUserDet d;
    d.x0 = x0, d.y0 = y0, d.x1 = x1, d.y1 = y1, d.score = 0.8f, d.cls = 2;
    for (float &p : d.p) p = 0.f;
    return d;
}

double plane_mean(const std::vector<float> &in, int k) {
    const size_t n = size_t(kPoseS) * kPoseS;
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += in[size_t(k) * n + i];
    return s / double(n);
}

std::unique_ptr<ModelRunner> load(const std::string &path) {
    RunnerOptions po;
    po.backend = "ort", po.model = path, po.threads = 1;
    po.input_shape = {1, 3 + kPoseG, kPoseS, kPoseS};
    po.output_size = kPoseOut;
    return make_runner(po);
}

int bench(const std::string &path, int n) {
    auto m = load(path);
    Frame f(1280, 720, 120, 128, 128);
    for (size_t i = 0; i < size_t(1280) * 720; ++i) f.d[i] = uint8_t(60 + (i * 2654435761u >> 24) % 120);
    std::vector<float> in(size_t(3 + kPoseG) * kPoseS * kPoseS), out(kPoseOut);
    const float sizes[3][2] = {{24, 18}, {70, 55}, {220, 160}};
    for (const auto &s : sizes) {
        RoadUserDet u = box(600, 300, 600 + s[0], 300 + s[1]);
        std::vector<double> t;
        RuPose p;
        for (int i = 0; i < n; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            ru_pose_input(f.view(), u, 0, PoseCam(), in.data());
            const auto t1 = std::chrono::steady_clock::now();
            m->run(in.data(), out.data());
            ru_pose_decode(out.data(), u, p);
            const auto t2 = std::chrono::steady_clock::now();
            t.push_back(std::chrono::duration<double, std::milli>(t2 - t0).count());
            if (i == 0) printf("box %3.0f x %3.0f: crop %.3f ms, ", double(s[0]), double(s[1]), std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(t.begin(), t.end());
        printf("per box median %.3f ms, p90 %.3f ms (%d runs)\n", t[t.size() / 2], t[t.size() * 9 / 10], n);
    }
    return 0;
}
}  // namespace

int main(int argc, char **argv) {
    if (argc >= 3 && !strcmp(argv[1], "--bench")) return bench(argv[2], argc > 3 ? atoi(argv[3]) : 2001);
    if (argc < 2) {
        fprintf(stderr, "usage: pose_test FIXTURE.onnx | --bench MODEL.onnx [N]\n");
        return 2;
    }
    std::vector<float> in(size_t(3 + kPoseG) * kPoseS * kPoseS, -1.f), out(kPoseOut, 0.f);
    // a grey frame (Y 100, neutral chroma): R = G = B = 100 / 255
    Frame f(640, 360, 100, 128, 128);
    const double grey = 100.0 / 255.0;

    {  // a box well inside: the whole crop is picture; the box numbers
        RoadUserDet u = box(200, 100, 300, 150);
        PoseCam cam;
        CHECK(ru_pose_input(f.view(), u, 0, cam, in.data()));
        for (int k = 0; k < 3; ++k) CHECK(near(plane_mean(in, k), grey, 0.004));
        CHECK(near(plane_mean(in, 3), std::log(2.0), 1e-5));
        for (int k = 4; k < 12; ++k) CHECK(near(plane_mean(in, k), 0.0, 1e-7));
        // the camera numbers
        cam.ok = true, cam.f = 500.f, cam.cu = 320.f, cam.cv = 180.f;
        CHECK(ru_pose_input(f.view(), u, 4 | 8, cam, in.data()));
        CHECK(near(plane_mean(in, 4), 0, 1e-7) && near(plane_mean(in, 5), 0, 1e-7));
        CHECK(near(plane_mean(in, 6), 1, 1e-7) && near(plane_mean(in, 7), 1, 1e-7));
        CHECK(near(plane_mean(in, 8), 1, 1e-7));
        CHECK(near(plane_mean(in, 9), (250.0 - 320.0) / 500.0, 1e-5));
        CHECK(near(plane_mean(in, 10), (150.0 - 180.0) / 500.0, 1e-5));
        CHECK(near(plane_mean(in, 11), std::log(50.0 / 500.0) + 3.0, 1e-5));
    }
    {  // a box at the frame's left edge: the crop's part outside the picture is black, the edge counts as cut
        RoadUserDet u = box(0, 100, 100, 200);  // crop x from -15 to 115: 15 of 130 outside = 7.4 of 64 columns
        CHECK(ru_pose_input(f.view(), u, 0, PoseCam(), in.data()));
        const double inside = plane_mean(in, 0) / grey;   // share of the crop that is picture
        CHECK(inside > 0.86 && inside < 0.89);            // 56 of 64 columns (whole crop pixels only)
        CHECK(near(plane_mean(in, 4), 1, 1e-7));          // left edge cut
        CHECK(near(in[0], 0.0, 1e-7));                    // top-left crop pixel: outside
        CHECK(near(in[size_t(kPoseS) * 32 + 40], grey, 0.004));
    }
    {  // a coloured frame: the crop is RGB in this order
        Frame c(640, 360, 110, 100, 190);   // reddish
        RoadUserDet u = box(200, 100, 300, 150);
        CHECK(ru_pose_input(c.view(), u, 0, PoseCam(), in.data()));
        const double r = (110 + 1.402 * 62) / 255.0, g = (110 - 0.344136 * -28 - 0.714136 * 62) / 255.0, b = (110 + 1.772 * -28) / 255.0;
        CHECK(near(plane_mean(in, 0), r, 0.006) && near(plane_mean(in, 1), g, 0.006) && near(plane_mean(in, 2), b, 0.006));
    }
    {  // too small
        RoadUserDet u = box(200, 100, 206, 150);
        CHECK(!ru_pose_input(f.view(), u, 0, PoseCam(), in.data()));
    }
    {  // decode: box units -> frame pixels
        RoadUserDet u = box(200, 100, 300, 150);
        float o[kPoseOut];
        for (int i = 0; i < kPoseOut; ++i) o[i] = 0.f;
        o[0] = -12.5f, o[1] = 6.f, o[2] = 0.9f, o[3] = 0.04f, o[4] = 0.8f, o[7] = 0.2f;
        o[8] = 0.1f, o[9] = -0.2f, o[10] = -0.02f, o[11] = -0.5f, o[12] = -0.3f, o[13] = 0.5f, o[14] = 0.0f;
        for (int i = 15; i < 22; ++i) o[i] = 0.05f;
        o[22] = 0.9f, o[23] = 1.2f, o[26] = 4.2f, o[27] = 1.7f, o[28] = 1.5f, o[35] = 1.f;
        RuPose p;
        CHECK(ru_pose_decode(o, u, p) && p.ok);
        CHECK(near(p.yaw, -12.5, 1e-5) && near(p.yaw_std, 6, 1e-5));
        CHECK(near(p.gc[0], 260, 1e-3) && near(p.gc[1], 140, 1e-3));
        CHECK(near(p.gn_y, 149, 1e-3));
        CHECK(near(p.gl[0], 200, 1e-3) && near(p.gl[1], 135, 1e-3) && near(p.gr[0], 300, 1e-3) && near(p.gr[1], 150, 1e-3));
        CHECK(near(p.gc_std[0], 5, 1e-3) && near(p.gc_std[1], 2.5, 1e-3) && near(p.gn_std, 2.5, 1e-3));
        CHECK(near(p.gn_vis, 1.0, 1e-6));   // probabilities are clamped
        const std::string j = pose_json(p);
        CHECK(j.find(",\"pose\":{\"yaw\":-12.5,\"yaw_std\":6.0,\"p\":0.90,\"flip\":0.04,\"vp\":\"rear\",\"vp_conf\":0.80,") == 0);
        CHECK(j.find("\"gc\":[260.0,140.0,5.0,2.5,0.90]") != std::string::npos);
        CHECK(j.find("\"gn\":[260.0,149.0,2.5,1.00]") != std::string::npos);
        CHECK(j.find("\"dim\":[4.20,1.70,1.50]") != std::string::npos);
        CHECK(j.back() == '}');
        o[5] = std::nanf("");
        CHECK(!ru_pose_decode(o, u, p) && !p.ok);
        CHECK(pose_json(p).empty());
        // a road user without a pose has no key; with one, the key is inside its object
        RoadUserDet v = box(200, 100, 300, 150);
        CHECK(road_user_json(v).find("pose") == std::string::npos);
        o[5] = 0.f;
        ru_pose_decode(o, v, v.pose);
        const std::string rj = road_user_json(v);
        CHECK(rj.find(",\"pose\":{") != std::string::npos && rj.back() == '}');
    }
    // the stand-in model through the runner: every output is a known function of the input (make_fixture.py)
    try {
        auto m = load(argv[1]);
        Frame c(640, 360, 110, 100, 190);
        RoadUserDet u = box(0, 100, 100, 150);
        PoseCam cam;
        cam.ok = true, cam.f = 500.f, cam.cu = 320.f, cam.cv = 180.f;
        CHECK(ru_pose_input(c.view(), u, 8, cam, in.data()));
        m->run(in.data(), out.data());
        CHECK(near(out[0], 100.0 * plane_mean(in, 0), 1e-3));
        CHECK(near(out[4], 1, 1e-5) && near(out[5], 0, 1e-5) && near(out[6], 0, 1e-5) && near(out[7], 1, 1e-5));
        CHECK(near(out[8], std::log(2.0), 1e-4));
        CHECK(near(out[9], (50.0 - 320.0) / 500.0, 1e-4) && near(out[10], (150.0 - 180.0) / 500.0, 1e-4));
        CHECK(near(out[11], std::log(50.0 / 500.0) + 3.0, 1e-4) && near(out[12], 1, 1e-5));
        CHECK(near(out[13], plane_mean(in, 1), 1e-4) && near(out[14], plane_mean(in, 2), 1e-4));
        RuPose p;
        CHECK(ru_pose_decode(out.data(), u, p));
        CHECK(near(p.gc[0], 50 + std::log(2.0) * 100, 0.02) && near(p.dim[0], 4.0, 1e-5) && near(p.kind[0], 1, 1e-6));
        // a model of another shape is refused
        bool threw = false;
        try {
            RunnerOptions bad;
            bad.backend = "ort", bad.model = argv[1], bad.threads = 1;
            bad.input_shape = {1, 3, 32, 32};
            bad.output_size = 32;
            make_runner(bad);
        } catch (const std::exception &) { threw = true; }
        CHECK(threw);
    } catch (const std::exception &e) {
        fprintf(stderr, "FAILED: %s\n", e.what());
        ++failures;
    }
    if (failures) {
        fprintf(stderr, "pose_test: %d check(s) failed\n", failures);
        return 1;
    }
    printf("pose_test: all checks passed\n");
    return 0;
}
