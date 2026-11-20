// lights_replay: run signd's traffic-light pipeline (signd/lights.h) offline on JPEG files or raw NV12
// sequences and print one `lights` JSON line per frame (with "raw" detections and "timing"), plus a timing
// summary on stderr. Used for the accuracy checks (C++ vs the laptop evaluation) and the phone cost table in
// roadsense docs/lanes/L7-traffic-lights.md.
//
//   lights_replay [options] FILE...
//     FILE  .jpg/.jpeg (libjpeg build; converted to NV12 like the ring holds it) or raw NV12 .nv12/.yuv
//           (needs --size WxH; several frames back to back are a sequence)
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifdef SIGND_HAVE_LIBJPEG
#include <csetjmp>
#include <jpeglib.h>
#endif

#include "lights.h"

using namespace roadsense::signs;
using Clock = std::chrono::steady_clock;

namespace {

bool ends_with(const std::string &s, const char *suf) {
    const size_t n = strlen(suf);
    if (s.size() < n) return false;
    for (size_t i = 0; i < n; ++i)
        if (tolower(uint8_t(s[s.size() - n + i])) != suf[i]) return false;
    return true;
}

bool read_all(const std::string &path, std::vector<uint8_t> &buf) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    buf.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

#ifdef SIGND_HAVE_LIBJPEG
struct JErr {
    jpeg_error_mgr pub;
    jmp_buf jb;
};
void jerr_exit(j_common_ptr c) { longjmp(reinterpret_cast<JErr *>(c->err)->jb, 1); }
bool read_jpeg(const std::string &path, std::vector<uint8_t> &px, int &w, int &h) {
    std::vector<uint8_t> b;
    if (!read_all(path, b)) return false;
    jpeg_decompress_struct cinfo;
    JErr jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jerr_exit;
    if (setjmp(jerr.jb)) {
        jpeg_destroy_decompress(&cinfo);
        return false;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, b.data(), b.size());
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);
    w = int(cinfo.output_width);
    h = int(cinfo.output_height);
    px.resize(size_t(w) * h * 3);
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW row = &px[size_t(cinfo.output_scanline) * w * 3];
        jpeg_read_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return true;
}
#endif

bool parse_floats(const std::string &s, float *out, int n) {
    const char *p = s.c_str();
    for (int i = 0; i < n; ++i) {
        char *e;
        out[i] = strtof(p, &e);
        if (e == p) return false;
        p = *e == ',' ? e + 1 : e;
    }
    return true;
}

void usage() {
    fprintf(stderr,
            "usage: lights_replay [options] FILE...\n"
            "  --backend ncnn|ncnn-cpu|ort (ncnn)  --model PATH  --models-dir DIR  --in WxH (960x320)\n"
            "  --roi x0,y0,x1,y1  fractions of the frame (0,0,1,0.62)   --thr X (0.25)  --nms X (0.45)\n"
            "  --state rule|PATH.onnx (rule)   --threads N (2)   --cpus LIST (none)   --ncnn-fp32  --ncnn-no-flushall\n"
            "  --roi2 x0,y0,x1,y1  second view on every other processed frame (none)   --thr-keep X (off)   --coast N (0)\n"
            "  --dt-ms X  time between processed frames for the tracker (200)\n"
            "  --road-users  also decode the road users (signd --lights-road-users) into each message\n"
            "  --both     with --roi2: run both views on every frame (detections merged by NMS) instead of alternating\n"
            "  --idle P   while the tracker holds no track at all, the frames follow the pattern P instead of wide, zoom,\n"
            "             wide, ...: one letter per frame, W = wide view, Z = second view, . = no detector pass (signd\n"
            "             --lights-idle P), e.g. Z.ZW; a skipped frame prints the message a consumer still holds (no lights)\n"
            "  --view-phase 1  start with the second view (zoom, wide, ...): the other phase of the same schedule\n"
            "  --idle-every N  = --idle with W and Z alternating on every Nth frame (N=2: W.Z.)\n"
            "  --size WxH (raw NV12)  --every N (1)  --max-frames N  --repeat N  --no-track  --bench N (time N more passes)\n");
}

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(q * (v.size() - 1) + 0.5))];
}

}  // namespace

int main(int argc, char **argv) {
    LightsConfig lc;
    LightTrackerConfig tc;
    std::vector<std::string> files;
    std::string cpus = "none";
    int raw_w = 0, raw_h = 0, every = 1, repeat = 1, bench = 0, idle_every = 1;
    std::string idle_pat;
    uint64_t view_phase = 0;  // 1: zoom, wide, zoom, ... (the other phase of the same schedule: a noise estimate)
    uint64_t max_frames = 0;
    bool track = true, both = false;
    double dt_ms = 200;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) usage(), exit(2);
            return argv[++i];
        };
        if (a == "--backend") lc.backend = val();
        else if (a == "--model") lc.model = val();
        else if (a == "--models-dir") lc.models_dir = val();
        else if (a == "--in") {
            if (sscanf(val().c_str(), "%dx%d", &lc.in_w, &lc.in_h) != 2) return usage(), 2;
        } else if (a == "--roi") {
            if (!parse_floats(val(), lc.roi, 4)) return usage(), 2;
        } else if (a == "--roi2") {
            if (!parse_floats(val(), lc.roi2, 4)) return usage(), 2;
        } else if (a == "--thr") lc.det_thr = float(atof(val().c_str()));
        else if (a == "--thr-keep") lc.det_thr_keep = float(atof(val().c_str()));
        else if (a == "--coast") tc.report_coast = atoi(val().c_str());
        else if (a == "--dt-ms") dt_ms = atof(val().c_str());
        else if (a == "--nms") lc.nms_iou = float(atof(val().c_str()));
        else if (a == "--state") lc.state = val();
        else if (a == "--threads") lc.threads = atoi(val().c_str());
        else if (a == "--cpus") cpus = val();
        else if (a == "--ncnn-fp32") lc.ncnn_fp16 = false;
        else if (a == "--ncnn-no-flushall") lc.ncnn_flushall = false;
        else if (a == "--size") {
            if (sscanf(val().c_str(), "%dx%d", &raw_w, &raw_h) != 2) return usage(), 2;
        } else if (a == "--every") every = std::max(1, atoi(val().c_str()));
        else if (a == "--max-frames") max_frames = strtoull(val().c_str(), nullptr, 10);
        else if (a == "--repeat") repeat = std::max(1, atoi(val().c_str()));
        else if (a == "--no-track") track = false;
        else if (a == "--idle-every") {
            idle_every = std::max(1, atoi(val().c_str()));
            idle_pat = "W" + std::string(size_t(idle_every - 1), '.') + "Z" + std::string(size_t(idle_every - 1), '.');
        } else if (a == "--idle") idle_pat = val();
        else if (a == "--view-phase") view_phase = uint64_t(std::max(0, atoi(val().c_str())));
        else if (a == "--both") both = true;
        else if (a == "--bench") bench = atoi(val().c_str());
        else if (a == "--road-users") lc.road_users = true;
        else if (a == "--help" || a == "-h") return usage(), 0;
        else if (!a.empty() && a[0] == '-') return usage(), 2;
        else files.push_back(a);
    }
    if (files.empty()) return usage(), 2;
    if (cpus != "none") {
        cpu_set_t set;
        CPU_ZERO(&set);
        const char *s = cpus.c_str();
        while (*s) {
            char *e;
            long a = strtol(s, &e, 10), b = a;
            if (*e == '-') b = strtol(e + 1, &e, 10);
            for (long c = a; c <= b; ++c) CPU_SET(int(c), &set);
            s = *e == ',' ? e + 1 : e;
            if (e == s && *s) break;
        }
        if (sched_setaffinity(0, sizeof(set), &set) != 0) perror("sched_setaffinity");
    }
    std::unique_ptr<LightPipeline> pipe;
    try {
        pipe = std::make_unique<LightPipeline>(lc);
    } catch (const std::exception &e) {
        fprintf(stderr, "lights_replay: %s\n", e.what());
        return 1;
    }
    fprintf(stderr, "lights_replay: %s on %s, input %dx%d, roi %.2f,%.2f,%.2f,%.2f, state %s\n", pipe->config().model.c_str(),
            pipe->backend(), lc.in_w, lc.in_h, lc.roi[0], lc.roi[1], lc.roi[2], lc.roi[3], lc.state.c_str());
    if (lc.det_thr_keep > 0) tc.new_score = lc.det_thr;
    LightTracker tracker(tc);
    std::vector<LightDet> dets;
    std::vector<LightTrackReport> reps;
    std::vector<double> t_pre, t_det, t_state, t_total, t_users;
    uint64_t seq = 0, passes = view_phase, idle_n = 0, skipped = 0;
    Nv12Image frame;
    auto run = [&](const Nv12View &v) {
        LightsTiming t;
        // --idle: nothing tracked (no track of any age, so nothing to age or coast) -> the idle pattern decides
        // whether this frame gets a detector pass and of which view. Time (seq) goes on either way.
        int view = pipe->has_view2() && !both ? int(passes & 1) : 0;   // wide, zoom, wide, ... like signd
        if (track && !idle_pat.empty() && tracker.live() == 0) {
            const char c = idle_pat[idle_n++ % idle_pat.size()];
            if (c == '.') {
                ++seq, ++skipped;
                const int64_t ts = int64_t(double(seq) * dt_ms * 1e6);
                dets.clear(), reps.clear();
                printf("%s\n", lights_json(seq, ts, 0, 0.0, v.width, v.height, pipe->config(), reps, &dets, &t, -1,
                                            nullptr, 0).c_str());
                return;
            }
            view = c == 'Z' && pipe->has_view2() ? 1 : 0;
            passes = uint64_t(view);  // the alternation goes on from this view when a track starts
        } else {
            idle_n = 0;  // the next idle stretch starts at the pattern's first letter
        }
        ++passes;
        pipe->process(v, dets, &t, view);
        if (both && pipe->has_view2()) {  // second view on the same frame, merged by score-ordered NMS
            std::vector<LightDet> d2;
            LightsTiming t2;
            pipe->process(v, d2, &t2, 1);
            t.det_ms += t2.det_ms, t.total_ms += t2.total_ms;
            dets.insert(dets.end(), d2.begin(), d2.end());
            std::stable_sort(dets.begin(), dets.end(), [](const LightDet &x, const LightDet &y) { return x.score > y.score; });
            std::vector<LightDet> kept;
            for (const LightDet &d : dets) {
                bool ok = true;
                for (const LightDet &k : kept) {
                    const float ix = std::max(0.f, std::min(d.x1, k.x1) - std::max(d.x0, k.x0));
                    const float iy = std::max(0.f, std::min(d.y1, k.y1) - std::max(d.y0, k.y0));
                    const float in = ix * iy;
                    if (in / ((d.x1 - d.x0) * (d.y1 - d.y0) + (k.x1 - k.x0) * (k.y1 - k.y0) - in + 1e-9f) > 0.45f) ok = false;
                }
                if (ok) kept.push_back(d);
            }
            dets.swap(kept);
        }
        ++seq;
        const int64_t ts = int64_t(double(seq) * dt_ms * 1e6);  // nominal 5 Hz for the tracker unless --dt-ms
        const float *vr = pipe->view_roi(view);
        const float seen[4] = {vr[0] * v.width, vr[1] * v.height, vr[2] * v.width, vr[3] * v.height};
        if (track) tracker.update(seq, ts, dets, reps, seen);
        else reps.clear();
        printf("%s\n", lights_json(seq, ts, 0, t.total_ms, v.width, v.height, pipe->config(), reps, &dets, &t, view,
                                    &pipe->road_users(), lc.road_users ? 1 : 0).c_str());
        t_users.push_back(t.users_ms);
        t_pre.push_back(t.pre_ms), t_det.push_back(t.det_ms), t_state.push_back(t.state_ms), t_total.push_back(t.total_ms);
    };
    for (const std::string &path : files) {
        for (int rep = 0; rep < repeat; ++rep) {
            if (ends_with(path, ".nv12") || ends_with(path, ".yuv")) {
                std::vector<uint8_t> b;
                const size_t fsz = size_t(raw_w) * raw_h * 3 / 2;
                if (!(raw_w > 0 && raw_h > 0 && read_all(path, b) && b.size() >= fsz)) {
                    fprintf(stderr, "lights_replay: cannot read %s (raw NV12 needs --size WxH)\n", path.c_str());
                    return 1;
                }
                for (size_t k = 0; (k + 1) * fsz <= b.size(); k += size_t(every)) {
                    Nv12View v;
                    v.y = b.data() + k * fsz;
                    v.uv = v.y + size_t(raw_w) * raw_h;
                    v.width = raw_w, v.height = raw_h, v.y_stride = v.uv_stride = raw_w;
                    run(v);
                    if (max_frames && seq >= max_frames) break;
                }
                frame.width = raw_w, frame.height = raw_h;
                frame.data.assign(b.end() - long(fsz), b.end());
            } else {
#ifdef SIGND_HAVE_LIBJPEG
                std::vector<uint8_t> px;
                int w, h;
                if (!read_jpeg(path, px, w, h)) {
                    fprintf(stderr, "lights_replay: cannot read %s\n", path.c_str());
                    continue;
                }
                rgb_to_nv12(px.data(), w, h, frame);
                if (!track) tracker.reset();
                run(frame.view());
#else
                fprintf(stderr, "lights_replay: no JPEG support in this build\n");
                return 1;
#endif
            }
            if (max_frames && seq >= max_frames) break;
        }
        if (max_frames && seq >= max_frames) break;
    }
    if (!idle_pat.empty())
        fprintf(stderr, "lights_replay: idle %s: %llu of %llu frames without a detector pass (%.1f %%)\n", idle_pat.c_str(),
                (unsigned long long)skipped, (unsigned long long)seq, seq ? 100.0 * double(skipped) / double(seq) : 0.0);
    fprintf(stderr, "lights_replay: %llu frames, median/p90 ms: pre %.1f/%.1f det %.1f/%.1f state %.2f/%.2f total %.1f/%.1f\n",
            (unsigned long long)seq, median(t_pre), pct(t_pre, 0.9), median(t_det), pct(t_det, 0.9), median(t_state),
            pct(t_state, 0.9), median(t_total), pct(t_total, 0.9));
    if (lc.road_users)
        fprintf(stderr, "lights_replay: road-user decode median/p90/max ms %.3f/%.3f/%.3f\n", median(t_users), pct(t_users, 0.9),
                t_users.empty() ? 0.0 : *std::max_element(t_users.begin(), t_users.end()));
    if (bench > 0 && !frame.data.empty()) {
        std::vector<double> d;
        for (int i = 0; i < bench; ++i) {
            LightsTiming t;
            pipe->process(frame.view(), dets, &t);
            d.push_back(t.det_ms);
        }
        fprintf(stderr, "lights_replay: bench %d passes: det median %.1f ms, p10 %.1f, p90 %.1f\n", bench, median(d), pct(d, 0.1),
                pct(d, 0.9));
    }
    return 0;
}
