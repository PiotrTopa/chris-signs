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
    int raw_w = 0, raw_h = 0, every = 1, repeat = 1, bench = 0;
    uint64_t max_frames = 0;
    bool track = true;
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
        } else if (a == "--thr") lc.det_thr = float(atof(val().c_str()));
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
        else if (a == "--bench") bench = atoi(val().c_str());
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
    LightTracker tracker(tc);
    std::vector<LightDet> dets;
    std::vector<LightTrackReport> reps;
    std::vector<double> t_pre, t_det, t_state, t_total;
    uint64_t seq = 0;
    Nv12Image frame;
    auto run = [&](const Nv12View &v) {
        LightsTiming t;
        pipe->process(v, dets, &t);
        ++seq;
        const int64_t ts = int64_t(seq) * 200000000;  // nominal 5 Hz for the tracker
        if (track) tracker.update(seq, ts, dets, reps);
        else reps.clear();
        printf("%s\n", lights_json(seq, ts, 0, t.total_ms, v.width, v.height, pipe->config(), reps, &dets, &t).c_str());
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
    fprintf(stderr, "lights_replay: %llu frames, median/p90 ms: pre %.1f/%.1f det %.1f/%.1f state %.2f/%.2f total %.1f/%.1f\n",
            (unsigned long long)seq, median(t_pre), pct(t_pre, 0.9), median(t_det), pct(t_det, 0.9), median(t_state),
            pct(t_state, 0.9), median(t_total), pct(t_total, 0.9));
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
