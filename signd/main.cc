// signd: camera frame ring -> MTSD sign detector + classifier (ONNX Runtime
// CPU) -> SORT tracker -> `signs` JSON lines on <rundir>/signd.sock and best
// crops in <rundir>/crops/<track_id>.jpg. Contract: roadsense docs/ARCHITECTURE.md,
// models and pre/post-processing: docs/SIGN_MODELS.md.
//
// Single thread (plus ORT's intra-op pool): take the NEWEST ring frame, run
// the whole pipeline, publish, repeat. Frames that arrive meanwhile are
// skipped, so the rate is simply whatever the cores give (2-4 fps on the
// A55 cluster, which is the default so modeld keeps the A75s).
#include <sched.h>
#include <signal.h>
#include <sys/resource.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#include "jsonsock.h"
#include "output.h"
#include "ring.h"
#include "tracker.h"
#include "yolo.h"

using namespace roadsense;
using namespace roadsense::signs;
using Clock = std::chrono::steady_clock;

namespace {

std::atomic<bool> g_run{true};
void on_signal(int) { g_run = false; }

struct Options {
    PipelineConfig pc;
    TrackerConfig tc;
    std::string cpus = "0-3";
    int nice = 0;
    size_t max_crops = 500;
    uint64_t max_frames = 0;
    double stats_s = 10;
    bool timing = false;
    bool verbose = false;
};

std::string config_dir() {
    const char *d = getenv("ROADSENSE_CONFIG_DIR");
    return d && *d ? d : "/etc/roadsense";
}

bool read_file(const std::string &path, std::string *out) {
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    *out = ss.str();
    return true;
}

// Minimal flat-JSON lookups (config files of scalars and strings).
size_t json_value_pos(const std::string &s, const char *key) {
    const std::string k = std::string("\"") + key + "\"";
    size_t p = s.find(k);
    if (p == std::string::npos) return p;
    p = s.find(':', p + k.size());
    if (p == std::string::npos) return p;
    ++p;
    while (p < s.size() && isspace(uint8_t(s[p]))) ++p;
    return p;
}
bool json_number(const std::string &s, const char *key, double *out) {
    size_t p = json_value_pos(s, key);
    if (p == std::string::npos) return false;
    if (s.compare(p, 4, "true") == 0) return *out = 1, true;
    if (s.compare(p, 5, "false") == 0) return *out = 0, true;
    char *end = nullptr;
    double v = strtod(s.c_str() + p, &end);
    if (end == s.c_str() + p) return false;
    *out = v;
    return true;
}
bool json_string(const std::string &s, const char *key, std::string *out) {
    size_t p = json_value_pos(s, key);
    if (p == std::string::npos || p >= s.size() || s[p] != '"') return false;
    size_t e = s.find('"', p + 1);
    if (e == std::string::npos) return false;
    *out = s.substr(p + 1, e - p - 1);
    return true;
}

bool parse_cpus(const std::string &s, cpu_set_t *set) {
    CPU_ZERO(set);
    size_t i = 0;
    int n = 0;
    while (i < s.size()) {
        char *end;
        long a = strtol(s.c_str() + i, &end, 10);
        if (end == s.c_str() + i) return false;
        i = size_t(end - s.c_str());
        long b = a;
        if (i < s.size() && s[i] == '-') {
            b = strtol(s.c_str() + i + 1, &end, 10);
            i = size_t(end - s.c_str());
        }
        for (long c = a; c <= b && c < CPU_SETSIZE; ++c) CPU_SET(int(c), set), ++n;
        if (i < s.size() && s[i] == ',') ++i;
    }
    return n > 0;
}

void set_models_dir(PipelineConfig &pc, const std::string &d) {
    pc.det_model = d + "/mtsd_signdet_yolov8n_384x640.onnx";
    pc.cls_model = d + "/mtsd_signcls_yolov8n_64.onnx";
    pc.cls_names = d + "/mtsd_signcls_classes.txt";
}

void usage() {
    fprintf(stderr,
            "usage: signd [options]\n"
            "  --models-dir DIR  directory with the default model file names (default /var/lib/roadsense/models)\n"
            "  --det PATH        detector ONNX (mtsd_signdet_yolov8n_384x640.onnx)\n"
            "  --cls PATH        classifier ONNX (mtsd_signcls_yolov8n_64.onnx)\n"
            "  --cls-names PATH  classifier class list (mtsd_signcls_classes.txt)\n"
            "  --threads N       ORT intra-op threads (default 4)\n"
            "  --cpus LIST       CPU affinity, e.g. 0-3 or 4-7 or 0,1 (default 0-3 = A55 cluster; none = no pinning)\n"
            "  --nice N          setpriority nice value (default 0)\n"
            "  --spin            let ORT threads busy-wait between ops (default off: slower on the SD845)\n"
            "  --det-thr X       detector score threshold (default 0.35)\n"
            "  --cls-gate X      classifier probability needed for a class vote (default 0.6)\n"
            "  --nms-iou X       NMS IoU (default 0.45)\n"
            "  --cls-margin X    relative box padding for the classifier crop (default 0)\n"
            "  --max-age N       tracker: processed frames without match before a track ends (default 10)\n"
            "  --min-hits N      tracker: hits before a track is reported (default 3)\n"
            "  --iou-match X     tracker: min IoU to associate (default 0.1)\n"
            "  --crop-margin X   relative margin of the saved best crop (default 0.15)\n"
            "  --max-crops N     keep at most N crop files written by this process (default 500, 0 = unlimited)\n"
            "  --timing          add a non-contract \"timing\" object to every message\n"
            "  --frames N        exit after N processed frames (testing)\n"
            "  --stats S         stderr stats interval in s (default 10, 0 = off)\n"
            "  --verbose\n"
            "config: $ROADSENSE_CONFIG_DIR (default /etc/roadsense)/signd.json, keys models_dir, det_model,\n"
            "        cls_model, cls_names, threads, cpus, nice, spin, det_thr, cls_gate, nms_iou, cls_margin,\n"
            "        max_age, min_hits, iou_match, crop_margin, max_crops (CLI wins)\n"
            "env: ROADSENSE_RUNDIR (default /run/roadsense), ROADSENSE_RING (default /dev/shm/roadsense-frames)\n");
}

void load_config(Options &o) {
    std::string s;
    if (!read_file(config_dir() + "/signd.json", &s)) return;
    double v;
    std::string str;
    if (json_string(s, "models_dir", &str)) set_models_dir(o.pc, str);
    if (json_string(s, "det_model", &str)) o.pc.det_model = str;
    if (json_string(s, "cls_model", &str)) o.pc.cls_model = str;
    if (json_string(s, "cls_names", &str)) o.pc.cls_names = str;
    if (json_number(s, "threads", &v)) o.pc.threads = int(v);
    if (json_string(s, "cpus", &str)) o.cpus = str;
    if (json_number(s, "nice", &v)) o.nice = int(v);
    if (json_number(s, "spin", &v)) o.pc.allow_spinning = v != 0;
    if (json_number(s, "det_thr", &v)) o.pc.det_thr = float(v);
    if (json_number(s, "cls_gate", &v)) o.pc.cls_gate = o.tc.cls_gate = float(v);
    if (json_number(s, "nms_iou", &v)) o.pc.nms_iou = float(v);
    if (json_number(s, "cls_margin", &v)) o.pc.cls_margin = float(v);
    if (json_number(s, "max_age", &v)) o.tc.max_age = int(v);
    if (json_number(s, "min_hits", &v)) o.tc.min_hits = int(v);
    if (json_number(s, "iou_match", &v)) o.tc.iou_match = float(v);
    if (json_number(s, "crop_margin", &v)) o.tc.crop_margin = float(v);
    if (json_number(s, "max_crops", &v)) o.max_crops = size_t(v);
}

}  // namespace

int main(int argc, char **argv) {
    Options opt;
    load_config(opt);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--models-dir") set_models_dir(opt.pc, val());
        else if (a == "--det") opt.pc.det_model = val();
        else if (a == "--cls") opt.pc.cls_model = val();
        else if (a == "--cls-names") opt.pc.cls_names = val();
        else if (a == "--threads") opt.pc.threads = atoi(val().c_str());
        else if (a == "--cpus") opt.cpus = val();
        else if (a == "--nice") opt.nice = atoi(val().c_str());
        else if (a == "--spin") opt.pc.allow_spinning = true;
        else if (a == "--no-spin") opt.pc.allow_spinning = false;
        else if (a == "--det-thr") opt.pc.det_thr = float(atof(val().c_str()));
        else if (a == "--cls-gate") opt.pc.cls_gate = opt.tc.cls_gate = float(atof(val().c_str()));
        else if (a == "--nms-iou") opt.pc.nms_iou = float(atof(val().c_str()));
        else if (a == "--cls-margin") opt.pc.cls_margin = float(atof(val().c_str()));
        else if (a == "--max-age") opt.tc.max_age = atoi(val().c_str());
        else if (a == "--min-hits") opt.tc.min_hits = atoi(val().c_str());
        else if (a == "--iou-match") opt.tc.iou_match = float(atof(val().c_str()));
        else if (a == "--crop-margin") opt.tc.crop_margin = float(atof(val().c_str()));
        else if (a == "--max-crops") opt.max_crops = size_t(atol(val().c_str()));
        else if (a == "--timing") opt.timing = true;
        else if (a == "--frames") opt.max_frames = strtoull(val().c_str(), nullptr, 10);
        else if (a == "--stats") opt.stats_s = atof(val().c_str());
        else if (a == "--verbose" || a == "-v") opt.verbose = true;
        else {
            usage();
            return a == "--help" || a == "-h" ? 0 : 2;
        }
    }
    const char *rd = getenv("ROADSENSE_RUNDIR");
    const std::string rundir = (rd && *rd) ? rd : "/run/roadsense";

    if (opt.cpus != "none" && !opt.cpus.empty()) {
        cpu_set_t set;
        if (!parse_cpus(opt.cpus, &set)) {
            fprintf(stderr, "signd: bad --cpus '%s'\n", opt.cpus.c_str());
            return 2;
        }
        // Before ORT creates its pool: the intra-op threads inherit the mask.
        if (sched_setaffinity(0, sizeof(set), &set) != 0)
            fprintf(stderr, "signd: sched_setaffinity(%s): %s\n", opt.cpus.c_str(), strerror(errno));
    }
    if (opt.nice != 0 && setpriority(PRIO_PROCESS, 0, opt.nice) != 0)
        fprintf(stderr, "signd: setpriority(%d): %s\n", opt.nice, strerror(errno));
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);

    std::unique_ptr<SignPipeline> pipe;
    try {
        pipe = std::make_unique<SignPipeline>(opt.pc);
    } catch (const std::exception &e) {
        fprintf(stderr, "signd: model load failed: %s\n", e.what());
        return 1;
    }
    fprintf(stderr,
            "signd: detector %dx%d, %zu classes (%zu merged), %d threads on cpus %s, det_thr %.2f cls_gate %.2f "
            "nms %.2f\n",
            pipe->input_w(), pipe->input_h(), pipe->names().raw.size(), pipe->names().merged.size(), opt.pc.threads,
            opt.cpus.c_str(), opt.pc.det_thr, opt.pc.cls_gate, opt.pc.nms_iou);

    if (!mkdir_p(rundir)) {
        fprintf(stderr, "signd: cannot create %s\n", rundir.c_str());
        return 1;
    }
    JsonLineServer srv;
    std::string err;
    if (!srv.listen(rundir + "/signd.sock", &err)) {
        fprintf(stderr, "signd: %s\n", err.c_str());
        return 1;
    }
    CropStore crops(rundir + "/crops", opt.max_crops);
    if (!crops.init(&err)) fprintf(stderr, "signd: %s (crops disabled)\n", err.c_str());

    Tracker tracker(opt.tc, pipe->names());
    RingReader rr(ring_path());
    FrameView fv;
    std::vector<Detection> dets;
    std::vector<TrackReport> reports;
    std::vector<EndedTrack> ended;
    std::vector<std::string> crop_paths;
    uint64_t processed = 0, last_seq = 0, skipped = 0;
    bool waiting_logged = false;
    auto stats_t0 = Clock::now();
    uint64_t stats_n = 0;
    double stats_ms = 0, stats_det = 0;
    FrameMeta last_meta;

    auto publish = [&](const FrameMeta &m, double ms, const Timing *t) {
        crop_paths.clear();
        for (const EndedTrack &e : ended) crop_paths.push_back(crops.save(e.id, e.crop));
        ExtraFields ex;
        if (opt.timing) ex.timing = t;
        srv.publish(signs_json(m, ms, reports, ended, crop_paths, ex));
    };

    while (g_run) {
        if (!rr.is_open() && !rr.open()) {
            if (!waiting_logged) fprintf(stderr, "signd: waiting for ring %s\n", ring_path().c_str());
            waiting_logged = true;
            srv.service();
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }
        if (!rr.next(fv, 200)) {
            srv.service();
            continue;
        }
        if (fv.seq < last_seq) {  // camerad restarted: the old tracks cannot continue
            reports.clear();
            ended.clear();
            tracker.flush(ended);
            if (!ended.empty()) publish(last_meta, 0.0, nullptr);
        } else if (last_seq) {
            skipped += fv.seq - last_seq - 1;
        }
        last_seq = fv.seq;
        if (fv.format != kFormatNV12 || (fv.width & 1) || (fv.height & 1)) {
            fprintf(stderr, "signd: unsupported frame %ux%u format %u\n", fv.width, fv.height, fv.format);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        Nv12View v;
        v.y = fv.y;
        v.uv = fv.uv;
        v.width = int(fv.width);
        v.height = int(fv.height);
        v.y_stride = int(fv.y_stride);
        v.uv_stride = int(fv.uv_stride);
        Timing t;
        try {
            pipe->process(v, dets, &t);
        } catch (const std::exception &e) {
            fprintf(stderr, "signd: inference failed: %s\n", e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        ended.clear();
        const auto tt0 = Clock::now();
        tracker.update(fv.seq, v, dets, reports, ended);
        const double detect_ms = t.total_ms + std::chrono::duration<double, std::milli>(Clock::now() - tt0).count();
        FrameMeta m;
        m.seq = fv.seq;
        m.ts_mono_ns = fv.ts_mono_ns;
        m.ts_real_ns = fv.ts_real_ns;
        last_meta = m;
        publish(m, detect_ms, &t);
        ++processed;
        ++stats_n;
        stats_ms += detect_ms;
        stats_det += t.det_ms;
        if (opt.verbose)
            fprintf(stderr, "signd: seq %llu: %zu raw, %zu reported, %zu ended, %.1f ms (pre %.1f det %.1f cls %.1f)\n",
                    (unsigned long long)fv.seq, dets.size(), reports.size(), ended.size(), detect_ms, t.pre_ms,
                    t.det_ms, t.cls_ms);
        const double el = std::chrono::duration<double>(Clock::now() - stats_t0).count();
        if (opt.stats_s > 0 && el >= opt.stats_s) {
            fprintf(stderr,
                    "signd: %.2f fps, detect_ms %.1f (det %.1f), live tracks %zu, frames skipped %llu, clients %zu\n",
                    stats_n / el, stats_ms / stats_n, stats_det / stats_n, tracker.live(),
                    (unsigned long long)skipped, srv.client_count());
            stats_t0 = Clock::now();
            stats_n = 0;
            stats_ms = stats_det = 0;
        }
        if (opt.max_frames && processed >= opt.max_frames) break;
    }
    // Shutdown: end the live tracks so their crops and ends are published.
    reports.clear();
    ended.clear();
    tracker.flush(ended);
    if (!ended.empty()) publish(last_meta, 0.0, nullptr);
    srv.close();
    fprintf(stderr, "signd: exit after %llu frames\n", (unsigned long long)processed);
    return 0;
}
