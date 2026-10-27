// signd_replay: runs the signd pipeline (same library: detector, classifier,
// tracker, JSON) on image files instead of the camera ring and prints one
// `signs` JSON line per processed frame.
//
// Inputs: PPM (P6) / PGM (P5), JPEG (if built with libjpeg), or raw NV12
// frames (*.nv12 / *.yuv, needs --size WxH; a file holding several frames
// back to back is a sequence, --every N takes every Nth). RGB inputs are converted to
// full-range BT.601 NV12 first (odd sizes cropped by one pixel), so the
// pipeline sees exactly what it gets from camerad.
//
// The files form one sequence (consecutive frames, seq 1, 2, ...) unless
// --reset-each is given; --repeat N feeds every file N times (a still image
// then behaves like a parked car, which confirms tracks after min_hits).
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifdef SIGND_HAVE_LIBJPEG
#include <csetjmp>
#include <jpeglib.h>
#endif

#include "jpeg_nv12.h"
#include "output.h"
#include "tracker.h"
#include "yolo.h"

using namespace roadsense;
using namespace roadsense::signs;
using Clock = std::chrono::steady_clock;

namespace {

struct RgbImage {
    int w = 0, h = 0;
    std::vector<uint8_t> px;  // RGB
};

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

bool read_pnm(const std::string &path, RgbImage &img) {
    std::vector<uint8_t> b;
    if (!read_all(path, b) || b.size() < 3 || b[0] != 'P' || (b[1] != '6' && b[1] != '5')) return false;
    const bool rgb = b[1] == '6';
    size_t p = 2;
    int vals[3], n = 0;
    while (n < 3 && p < b.size()) {
        if (isspace(b[p])) {
            ++p;
        } else if (b[p] == '#') {
            while (p < b.size() && b[p] != '\n') ++p;
        } else {
            int v = 0;
            while (p < b.size() && isdigit(b[p])) v = v * 10 + (b[p++] - '0');
            vals[n++] = v;
        }
    }
    ++p;  // single whitespace after maxval
    if (n < 3 || vals[2] != 255) return false;
    img.w = vals[0];
    img.h = vals[1];
    const size_t need = size_t(img.w) * img.h * (rgb ? 3 : 1);
    if (b.size() < p + need) return false;
    img.px.resize(size_t(img.w) * img.h * 3);
    for (size_t i = 0; i < size_t(img.w) * img.h; ++i)
        for (int c = 0; c < 3; ++c) img.px[i * 3 + c] = rgb ? b[p + i * 3 + c] : b[p + i];
    return true;
}

#ifdef SIGND_HAVE_LIBJPEG
struct JErr {
    jpeg_error_mgr pub;
    jmp_buf jb;
};
void jerr_exit(j_common_ptr c) { longjmp(reinterpret_cast<JErr *>(c->err)->jb, 1); }

bool read_jpeg(const std::string &path, RgbImage &img) {
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
    img.w = int(cinfo.output_width);
    img.h = int(cinfo.output_height);
    img.px.resize(size_t(img.w) * img.h * 3);
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW row = &img.px[size_t(cinfo.output_scanline) * img.w * 3];
        jpeg_read_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return true;
}
#endif

// ------------------------------------------------------------- drawing ---
// 5x7 bitmap font for the characters that occur in MTSD labels and numbers.
const uint8_t *glyph(char c) {
    static const uint8_t digits[10][7] = {
        {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}, {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},
        {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}, {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E},
        {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}, {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E},
        {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}, {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},
        {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}, {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}};
    static const uint8_t lower[26][7] = {
        {0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F}, {0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x1E},
        {0x00, 0x00, 0x0E, 0x10, 0x10, 0x11, 0x0E}, {0x01, 0x01, 0x0D, 0x13, 0x11, 0x11, 0x0F},
        {0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E}, {0x06, 0x09, 0x08, 0x1C, 0x08, 0x08, 0x08},
        {0x00, 0x0F, 0x11, 0x11, 0x0F, 0x01, 0x0E}, {0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x11},
        {0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E}, {0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0C},
        {0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12}, {0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E},
        {0x00, 0x00, 0x1A, 0x15, 0x15, 0x11, 0x11}, {0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11},
        {0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E}, {0x00, 0x00, 0x1E, 0x11, 0x1E, 0x10, 0x10},
        {0x00, 0x00, 0x0D, 0x13, 0x0F, 0x01, 0x01}, {0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10},
        {0x00, 0x00, 0x0E, 0x10, 0x0E, 0x01, 0x1E}, {0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06},
        {0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D}, {0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04},
        {0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A}, {0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11},
        {0x00, 0x00, 0x11, 0x11, 0x0F, 0x01, 0x0E}, {0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F}};
    static const uint8_t dash[7] = {0, 0, 0, 0x1F, 0, 0, 0}, dot[7] = {0, 0, 0, 0, 0, 0x0C, 0x0C},
                         slash[7] = {0, 0x01, 0x02, 0x04, 0x08, 0x10, 0}, colon[7] = {0, 0x0C, 0x0C, 0, 0x0C, 0x0C, 0},
                         hash[7] = {0x0A, 0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x0A}, blank[7] = {0};
    if (c >= '0' && c <= '9') return digits[c - '0'];
    if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    if (c >= 'a' && c <= 'z') return lower[c - 'a'];
    switch (c) {
    case '-': return dash;
    case '.': return dot;
    case '/': return slash;
    case ':': return colon;
    case '#': return hash;
    default: return blank;
    }
}

void fill(RgbImage &im, int x0, int y0, int x1, int y1, const uint8_t c[3]) {
    x0 = std::max(0, x0);
    y0 = std::max(0, y0);
    x1 = std::min(im.w, x1);
    y1 = std::min(im.h, y1);
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) std::memcpy(&im.px[(size_t(y) * im.w + x) * 3], c, 3);
}

void rect(RgbImage &im, int x0, int y0, int x1, int y1, int t, const uint8_t c[3]) {
    fill(im, x0 - t, y0 - t, x1 + t, y0, c);
    fill(im, x0 - t, y1, x1 + t, y1 + t, c);
    fill(im, x0 - t, y0, x0, y1, c);
    fill(im, x1, y0, x1 + t, y1, c);
}

void text(RgbImage &im, int x, int y, const std::string &s, int sc, const uint8_t fg[3], const uint8_t bg[3]) {
    const int cw = 6 * sc, chh = 9 * sc;
    if (x + int(s.size()) * cw > im.w) x = std::max(0, im.w - int(s.size()) * cw);
    if (y < 0) y = 0;
    fill(im, x, y, x + int(s.size()) * cw, y + chh, bg);
    for (size_t i = 0; i < s.size(); ++i) {
        const uint8_t *g = glyph(s[i]);
        for (int r = 0; r < 7; ++r)
            for (int c = 0; c < 5; ++c)
                if (g[r] & (0x10 >> c))
                    fill(im, x + int(i) * cw + c * sc, y + sc + r * sc, x + int(i) * cw + (c + 1) * sc,
                         y + sc + (r + 1) * sc, fg);
    }
}

bool write_ppm(const std::string &path, const RgbImage &im) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return false;
    fprintf(f, "P6\n%d %d\n255\n", im.w, im.h);
    fwrite(im.px.data(), 1, im.px.size(), f);
    return fclose(f) == 0;
}

std::string basename_noext(const std::string &p) {
    std::string b = p.substr(p.find_last_of('/') + 1);
    const size_t d = b.find_last_of('.');
    return d == std::string::npos ? b : b.substr(0, d);
}

void usage() {
    fprintf(stderr,
            "usage: signd_replay [options] FILE...\n"
            "  FILE              .ppm/.pgm, .jpg/.jpeg (libjpeg build), or raw NV12 .nv12/.yuv (needs --size)\n"
            "  --size WxH        size of raw NV12 inputs\n"
            "  --backend ort|ncnn (ort)  --det-backend B  --cls-backend B  --ncnn-fp32  --ncnn-no-flushall\n"
            "  --models-dir DIR  (default /var/lib/roadsense/models)  --det/--cls/--cls-names PATH\n"
            "  --threads N (4)   --cpus LIST (none)   --spin\n"
            "  --every N         raw NV12 sequences: process every Nth frame (default 1)\n"
            "  --max-frames N    stop after N processed frames (default: all)\n"
            "  --det-thr X (0.35) --cls-gate X (0.6) --nms-iou X (0.45) --cls-margin X (0)\n"
            "  --max-age N (10)  --min-hits N (3)  --iou-match X (0.1)  --crop-margin X (0.15)\n"
            "  --dist-match X (1.0)  --dist-match-new X (1.0, tracks with one hit)\n"
            "  --no-appearance   skip the glow check on round red detections (default on)\n"
            "  --verifier PATH   sign/lamp verifier .onnx (32x32 RGB -> logit) for round red detections (default none)\n"
            "  --glow-px X (0.10) --glow-share X (0.5) --verify-thr X (0.3) --verify-vouch X (0.9)   round red track\n"
            "                    suppression: verifier mean < thr, or glowing and verifier mean < vouch\n"
            "  --glow-alone      glowing tracks are suppressed without a verifier too (default: only with one)\n"
            "  --class-min-score LIST  prefix=min best detector score,... (e.g. regulatory--no-entry=0.7)\n"
            "  --repeat N        feed every file N times (default 1)\n"
            "  --reset-each      end all tracks after each file (default: files are one sequence)\n"
            "  --raw             add the per-frame detections before tracking (\"raw\", not in the contract)\n"
            "  --no-timing       omit the \"timing\" object\n"
            "  --draw DIR        write DIR/<name>.jpg with the raw detections drawn (--draw-ppm: .ppm)\n"
            "  --crops DIR       save ended-track best crops there (default: not saved)\n"
            "  --bench N         after the run, time N more detector passes on the last frame (median)\n"
            "stderr at the end: per-frame timing summary (pre/det/cls/total ms: median and p90) over the run\n");
}

}  // namespace

int main(int argc, char **argv) {
    PipelineConfig pc;
    TrackerConfig tc;
    std::string cpus = "none", draw_dir, crops_dir;
    int repeat = 1, bench = 0, raw_w = 0, raw_h = 0, every = 1;
    uint64_t max_frames = 0;
    bool reset_each = false, raw = false, timing = true, draw_ppm = false;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--models-dir") pc.models_dir = val();
        else if (a == "--backend") pc.det_backend = pc.cls_backend = val();
        else if (a == "--det-backend") pc.det_backend = val();
        else if (a == "--cls-backend") pc.cls_backend = val();
        else if (a == "--ncnn-fp32") pc.ncnn_fp16 = false;
        else if (a == "--ncnn-no-flushall") pc.ncnn_flushall = false;
        else if (a == "--every") every = std::max(1, atoi(val().c_str()));
        else if (a == "--max-frames") max_frames = strtoull(val().c_str(), nullptr, 10);
        else if (a == "--det") pc.det_model = val();
        else if (a == "--cls") pc.cls_model = val();
        else if (a == "--cls-names") pc.cls_names = val();
        else if (a == "--threads") pc.threads = atoi(val().c_str());
        else if (a == "--cpus") cpus = val();
        else if (a == "--spin") pc.allow_spinning = true;
        else if (a == "--no-spin") pc.allow_spinning = false;
        else if (a == "--det-thr") pc.det_thr = float(atof(val().c_str()));
        else if (a == "--cls-gate") pc.cls_gate = tc.cls_gate = float(atof(val().c_str()));
        else if (a == "--nms-iou") pc.nms_iou = float(atof(val().c_str()));
        else if (a == "--cls-margin") pc.cls_margin = float(atof(val().c_str()));
        else if (a == "--max-age") tc.max_age = atoi(val().c_str());
        else if (a == "--min-hits") tc.min_hits = atoi(val().c_str());
        else if (a == "--iou-match") tc.iou_match = float(atof(val().c_str()));
        else if (a == "--dist-match") tc.dist_match = float(atof(val().c_str()));
        else if (a == "--dist-match-new") tc.dist_match_new = float(atof(val().c_str()));
        else if (a == "--crop-margin") tc.crop_margin = float(atof(val().c_str()));
        else if (a == "--no-appearance") pc.appearance = false;
        else if (a == "--verifier") pc.verifier_model = val();
        else if (a == "--glow-px") tc.glow_px = float(atof(val().c_str()));
        else if (a == "--glow-share") tc.glow_share = float(atof(val().c_str()));
        else if (a == "--glow-alone") tc.glow_alone = true;
        else if (a == "--verify-vouch") tc.verify_vouch = float(atof(val().c_str()));
        else if (a == "--verify-thr") tc.verify_thr = float(atof(val().c_str()));
        else if (a == "--class-min-score") {
            if (!parse_class_min_score(val(), &tc.class_min_score)) {
                fprintf(stderr, "bad --class-min-score (prefix=score,...)\n");
                return 2;
            }
        }
        else if (a == "--repeat") repeat = std::max(1, atoi(val().c_str()));
        else if (a == "--reset-each") reset_each = true;
        else if (a == "--raw") raw = true;
        else if (a == "--no-timing") timing = false;
        else if (a == "--draw") draw_dir = val();
        else if (a == "--draw-ppm") draw_ppm = true;
        else if (a == "--crops") crops_dir = val();
        else if (a == "--bench") bench = atoi(val().c_str());
        else if (a == "--size") {
            if (sscanf(val().c_str(), "%dx%d", &raw_w, &raw_h) != 2) {
                usage();
                return 2;
            }
        } else if (a == "--help" || a == "-h") {
            usage();
            return 0;
        } else if (!a.empty() && a[0] == '-') {
            usage();
            return 2;
        } else files.push_back(a);
    }
    if (files.empty()) {
        usage();
        return 2;
    }
    if (cpus != "none") {
        cpu_set_t set;
        CPU_ZERO(&set);
        // same syntax as signd --cpus: a-b,c
        const char *s = cpus.c_str();
        while (*s) {
            char *e;
            long a = strtol(s, &e, 10), b = a;
            if (e == s) break;
            s = e;
            if (*s == '-') b = strtol(s + 1, &e, 10), s = e;
            for (long c = a; c <= b; ++c) CPU_SET(int(c), &set);
            if (*s == ',') ++s;
        }
        if (sched_setaffinity(0, sizeof(set), &set) != 0) perror("sched_setaffinity");
    }

    std::unique_ptr<SignPipeline> pipe;
    try {
        pipe = std::make_unique<SignPipeline>(pc);
    } catch (const std::exception &e) {
        fprintf(stderr, "signd_replay: %s\n", e.what());
        return 1;
    }
    Tracker tracker(tc, pipe->names());
    std::unique_ptr<CropStore> crops;
    if (!crops_dir.empty()) {
        crops = std::make_unique<CropStore>(crops_dir, 0);
        std::string err;
        if (!crops->init(&err)) fprintf(stderr, "signd_replay: %s\n", err.c_str());
    }
    if (!draw_dir.empty()) mkdir_p(draw_dir);

    std::vector<Detection> dets;
    std::vector<TrackReport> reports;
    std::vector<EndedTrack> ended;
    std::vector<std::string> crop_paths;
    Nv12Image frame;
    uint64_t seq = 0;
    FrameMeta last;
    auto emit = [&](const FrameMeta &m, double ms, const Timing *t, const std::vector<Detection> *rawd) {
        crop_paths.clear();
        for (const EndedTrack &e : ended) crop_paths.push_back(crops ? crops->save(e.id, e.crop) : std::string());
        ExtraFields ex;
        if (timing) ex.timing = t;
        if (raw && rawd) {
            ex.raw = rawd;
            ex.names = &pipe->names();
        }
        printf("%s\n", signs_json(m, ms, reports, ended, crop_paths, ex).c_str());
        fflush(stdout);
    };

    std::vector<double> t_pre, t_det, t_cls, t_tot;
    uint64_t processed = 0;
    bool stop = false;
    fprintf(stderr, "signd_replay: detector on %s (%s), classifier on %s (%s)\n", pipe->det_backend(),
            pipe->config().det_model.c_str(), pipe->cls_backend(), pipe->config().cls_model.c_str());
    auto run_frame = [&](const std::string &path, RgbImage &rgb) {
        const Nv12View v = frame.view();
        std::vector<Detection> first_dets;
        for (int r = 0; r < repeat; ++r) {
            ++seq;
            Timing t;
            pipe->process(v, dets, &t);
            if (r == 0) first_dets = dets;
            ended.clear();
            const auto tt0 = Clock::now();
            tracker.update(seq, v, dets, reports, ended);
            const double ms = t.total_ms + std::chrono::duration<double, std::milli>(Clock::now() - tt0).count();
            t_pre.push_back(t.pre_ms);
            t_det.push_back(t.det_ms);
            t_cls.push_back(t.cls_ms);
            t_tot.push_back(t.total_ms);
            FrameMeta m;
            m.seq = seq;
            last = m;
            emit(m, ms, &t, &dets);
            if (max_frames && ++processed >= max_frames) {
                stop = true;
                break;
            }
        }
        if (reset_each) {
            reports.clear();
            ended.clear();
            tracker.flush(ended);
            if (!ended.empty()) emit(last, 0.0, nullptr, nullptr);
        }
        if (!draw_dir.empty() && !rgb.px.empty()) {
            static const uint8_t mag[3] = {255, 0, 255}, white[3] = {255, 255, 255};
            const int sc = std::max(2, rgb.w / 640);
            for (const Detection &d : first_dets) {
                rect(rgb, int(d.x0), int(d.y0), int(d.x1), int(d.y1), 3, mag);
                char buf[32];
                snprintf(buf, sizeof(buf), " %.2f/%.2f", d.score, d.cls_prob);
                const std::string lab =
                    (d.cls_raw >= 0 ? strip_variant(pipe->names().raw[size_t(d.cls_raw)]) : std::string("sign")) + buf;
                text(rgb, int(d.x0), int(d.y0) - 9 * sc - 3, lab, sc, white, mag);
            }
            const std::string out = draw_dir + "/" + basename_noext(path) + (draw_ppm ? ".ppm" : ".jpg");
            bool wok;
            if (draw_ppm) {
                wok = write_ppm(out, rgb);
            } else {
                Nv12Image o;
                rgb_to_nv12(rgb.px.data(), rgb.w, rgb.h, o);
                JpegNV12Encoder enc(88);
                const Nv12View ov = o.view();
                const auto &j = enc.encode(ov.y, size_t(ov.y_stride), ov.uv, size_t(ov.uv_stride), o.width, o.height);
                wok = write_file_atomic(out, j.data(), j.size());
            }
            if (!wok) fprintf(stderr, "signd_replay: cannot write %s\n", out.c_str());
        }
    };

    for (const std::string &path : files) {
        if (stop) break;
        RgbImage rgb;
        if (ends_with(path, ".nv12") || ends_with(path, ".yuv")) {
            std::vector<uint8_t> b;
            const size_t fsz = size_t(raw_w) * raw_h * 3 / 2;
            if (!(raw_w > 0 && raw_h > 0 && read_all(path, b) && b.size() >= fsz)) {
                fprintf(stderr, "signd_replay: cannot read %s (raw NV12 needs --size WxH)\n", path.c_str());
                continue;
            }
            const size_t nframes = b.size() / fsz;
            fprintf(stderr, "signd_replay: %s %dx%d, %zu frame(s), every %d\n", path.c_str(), raw_w, raw_h, nframes, every);
            for (size_t k = 0; k < nframes && !stop; k += size_t(every)) {
                frame.width = raw_w;
                frame.height = raw_h;
                frame.data.assign(b.begin() + long(k * fsz), b.begin() + long((k + 1) * fsz));
                rgb.px.clear();
                if (!draw_dir.empty()) {
                    nv12_to_rgb(frame.view(), rgb.px);
                    rgb.w = raw_w;
                    rgb.h = raw_h;
                }
                run_frame(nframes > 1 ? path + "." + std::to_string(k) : path, rgb);
            }
            continue;
        }
        bool ok = false;
        if (ends_with(path, ".ppm") || ends_with(path, ".pgm") || ends_with(path, ".pnm")) {
            ok = read_pnm(path, rgb);
        } else if (ends_with(path, ".jpg") || ends_with(path, ".jpeg")) {
#ifdef SIGND_HAVE_LIBJPEG
            ok = read_jpeg(path, rgb);
#else
            fprintf(stderr, "signd_replay: built without libjpeg, convert %s to PPM\n", path.c_str());
#endif
        }
        if (!ok) {
            fprintf(stderr, "signd_replay: cannot read %s\n", path.c_str());
            continue;
        }
        rgb_to_nv12(rgb.px.data(), rgb.w, rgb.h, frame);
        fprintf(stderr, "signd_replay: %s %dx%d\n", path.c_str(), frame.width, frame.height);
        run_frame(path, rgb);
    }
    reports.clear();
    ended.clear();
    tracker.flush(ended);
    if (!ended.empty()) emit(last, 0.0, nullptr, nullptr);

    if (!t_tot.empty()) {
        auto q = [](std::vector<double> v, double f) {
            std::sort(v.begin(), v.end());
            return v[std::min(v.size() - 1, size_t(v.size() * f))];
        };
        fprintf(stderr,
                "signd_replay: %zu frames: pre %.1f/%.1f ms, det %.1f/%.1f ms, cls %.1f/%.1f ms, total %.1f/%.1f ms (median/p90)\n",
                t_tot.size(), q(t_pre, .5), q(t_pre, .9), q(t_det, .5), q(t_det, .9), q(t_cls, .5), q(t_cls, .9), q(t_tot, .5),
                q(t_tot, .9));
    }
    if (bench > 0 && frame.width > 0) {
        std::vector<double> tot, det;
        Timing t;
        for (int i = 0; i < bench; ++i) {
            pipe->process(frame.view(), dets, &t);
            tot.push_back(t.total_ms);
            det.push_back(t.det_ms);
        }
        std::sort(tot.begin(), tot.end());
        std::sort(det.begin(), det.end());
        fprintf(stderr, "signd_replay: bench %d runs (%zu boxes): pipeline median %.1f ms p90 %.1f, detector median %.1f ms p90 %.1f\n",
                bench, dets.size(), tot[tot.size() / 2], tot[tot.size() * 9 / 10], det[det.size() / 2],
                det[det.size() * 9 / 10]);
    }
    return 0;
}
