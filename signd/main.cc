// signd: camera frame ring -> MTSD sign detector + classifier (ONNX Runtime
// CPU, or ncnn/Vulkan with --backend ncnn) -> SORT tracker -> `signs` JSON lines on <rundir>/signd.sock and best
// crops in <rundir>/crops/<track_id>.jpg. Contract: roadsense docs/ARCHITECTURE.md,
// models and pre/post-processing: docs/SIGN_MODELS.md.
// With --lights a second thread runs the traffic-light pipeline (signd/lights.h, docs/LIGHT_MODELS.md) on its
// own ring reader at --lights-hz and serves `lights` JSON lines on <rundir>/lightd.sock; on the GPU it takes
// its own turns from modeld, serialised with the sign detector by a process mutex.
//
// Single thread (plus ORT's intra-op pool): take the NEWEST ring frame, run
// the whole pipeline, publish, repeat. Frames that arrive meanwhile are
// skipped, so the rate is simply whatever the cores give (2-4 fps on the
// A55 cluster, which is the default so modeld keeps the A75s).
#include <sched.h>
#include <signal.h>
#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <pthread.h>
#include <sstream>
#include <string>
#include <thread>

#include "governor.h"
#include "gpusched.h"
#include "jsonsock.h"
#include "lights.h"
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
    double max_hz = 0;  // 0 = as fast as the pipeline goes
    bool timing = false;
    bool verbose = false;
    std::string gpu_sync = "none";  // none | after | turn (roadsense docs/ARCHITECTURE.md, GPU schedule)
    std::string gpu_sched;          // default <rundir>/gpu.sched
    int gpu_sync_timeout_ms = 150;
    // traffic lights (second thread)
    bool lights = false;
    LightsConfig lc;
    LightTrackerConfig ltc;
    double lights_hz = 3;
    // While the light tracker holds no track at all (nothing to age, coast or re-state), the light frames follow
    // this pattern instead of wide, zoom, wide, ...: one letter per frame period of --lights-hz, W = wide view,
    // Z = second view, . = no detector pass. The first box brings the full schedule back with the next frame.
    // Empty = off. A demand file (`<rundir>/demand`, key `lights_idle`, same format and validity rule as the
    // governor's caps file) can forbid it while valid: 0 = never idle (the integrator, which knows the map and
    // the car, writes it where signals are expected), 1 or absent = idle allowed.
    std::string lights_idle;
    std::string lights_cpus;        // empty: same as --cpus
    bool lights_timing = false;
    bool lights_signs = false;      // --lights-signs: the light detector's `traffic sign` boxes go to the sign tracker
};

// One light frame's sign proposals, classified, waiting for the sign thread (--lights-signs).
struct SignProposals {
    uint64_t seq;
    int64_t ts_mono_ns, ts_real_ns;
    std::vector<roadsense::signs::Detection> dets;
    float seen[4];
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

void usage() {
    fprintf(stderr,
            "usage: signd [options]\n"
            "  --backend B       ort: ONNX Runtime on the CPU (default); ncnn: ncnn on the GPU through Vulkan (if built in)\n"
            "  --det-backend B, --cls-backend B   per-model override (e.g. detector on ncnn, classifier on ort)\n"
            "  --models-dir DIR  directory with the default model file names (default /var/lib/roadsense/models)\n"
            "  --det PATH        detector: ort mtsd_signdet_yolov8n_384x640.onnx, ncnn ...-ncnn.param (weights: .bin next to it)\n"
            "  --cls PATH        classifier: ort mtsd_signcls_yolov8n_64.onnx, ncnn ...-ncnn.param\n"
            "  --cls-names PATH  classifier class list (mtsd_signcls_classes.txt)\n"
            "  --threads N       ORT intra-op threads (default 4; ncnn needs none, use 1)\n"
            "  --ncnn-fp32       ncnn: disable fp16 packing (GPU fp32, several times slower)\n"
            "  --ncnn-no-flushall  ncnn: do not add TU_DEBUG=flushall (turnip needs it for correct results)\n"
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
            "  --dist-match X    tracker: max centre distance / box size, fallback pass (default 1.0)\n"
            "  --dist-match-new X  the same for tracks with one hit (default 1.0)\n"
            "  --crop-margin X   relative margin of the saved best crop (default 0.15)\n"
            "  --roi2 x0,y0,x1,y1  second detector view (frame fractions) on every --roi2-every-th frame, at the scale\n"
            "                    that fits it into the detector input (default none; e.g. 0.5,0.2,1,0.733 = the right\n"
            "                    640x384 px of a 1280x720 frame at scale 1.0)   --roi2-every N (2)\n"
            "  --no-appearance   skip the glow check on round red detections (default on)\n"
            "  --verifier PATH   sign/lamp verifier .onnx (32x32 RGB -> logit) for round red detections (default none)\n"
            "  --glow-px X (0.10) --glow-share X (0.5) --verify-thr X (0.3) --verify-vouch X (0.9)   round red track\n"
            "                    suppression: verifier mean < thr, or glowing and verifier mean < vouch\n"
            "  --glow-alone      glowing tracks are suppressed without a verifier too (default: only with one)\n"
            "  --class-min-score LIST  prefix=min best detector score,... (e.g. regulatory--no-entry=0.7)\n"
            "  --max-crops N     keep at most N crop files written by this process (default 500, 0 = unlimited)\n"
            "  --max-hz X        process at most X frames/s (duty cycle; default 0 = as fast as possible)\n"
            "  --gpu-sync M      none (default): ignore modeld; after: start each detector run right after a modeld\n"
            "                    run ends (modeld's GPU tick); turn: also ask modeld to wait until the detector is done\n"
            "  --gpu-sched PATH  GPU schedule file (default <rundir>/gpu.sched, created by modeld)\n"
            "  --gpu-sync-timeout MS  longest wait for modeld's tick (default 150); no wait if modeld is absent\n"
            "  --timing          add a non-contract \"timing\" object to every message\n"
            "  --lights          also detect traffic lights (thread of its own, <rundir>/lightd.sock, `lights` topic)\n"
            "  --lights-backend B  ncnn (GPU, default), ncnn-cpu or ort  --lights-model PATH  --lights-in WxH (960x320)\n"
            "  --lights-roi x0,y0,x1,y1  frame fractions fed to the detector (default 0,0,1,0.62)\n"
            "  --lights-roi2 x0,y0,x1,y1  second view on every other light frame (far lights at a higher scale; default none)\n"
            "  --lights-thr-keep X  boxes down to this score continue a track, never start one (default: off)\n"
            "  --lights-coast N  report a confirmed track for N more light frames after its last match (default 0)\n"
            "  --lights-hz X     light frames per second (default 3)  --lights-threads N (2)  --lights-cpus LIST\n"
            "  --lights-idle P   while no light is tracked at all, light frames follow the pattern P (one letter per frame\n"
            "                    period: W wide view, Z second view, . no detector pass), e.g. Z.ZW; default off.\n"
            "                    <rundir>/demand key lights_idle = 0 forbids it while valid (the integrator's hint)\n"
            "  --lights-signs    the light detector's `traffic sign` boxes (BDD100K class 9, score > --lights-signs-thr,\n"
            "                    default 0.35) are classified by the sign classifier and fed to the sign tracker as\n"
            "                    updates of their own (a `signs` message each, view 2, when not older than the last\n"
            "                    message); default off. --lights-signs-keep: the same, but those boxes only continue\n"
            "                    tracks the sign detector started; --lights-signs-nohits: those boxes are not counted\n"
            "                    as hits (they keep a track alive and placed and vote)\n"
            "  --lights-road-users  also publish the light detector's road users (person, rider, car, truck, bus, train,\n"
            "                    motor, bike; riders paired with their vehicle) as `road_users` in every `lights`\n"
            "                    message: no extra inference, paused by the governor's `road_users_hz` cap (\"road_users\":\n"
            "                    null while shed); --lights-ru-thr LIST: per-class score thresholds, `name=x,...` or 8 numbers\n"
            "  --lights-ru-roi x0,y0,x1,y1  with --lights-road-users: an EXTRA detector run on this part of the frame for\n"
            "                    road users only (a `road_users` message, view 2), after the light frame, at most\n"
            "                    --lights-ru-hz X a second (default 1); costs one detector run each; off by default\n"
            "  --lights-thr X (0.25)  --lights-state rule|PATH.onnx[,PATH2.onnx...] (rule; of a list the first\n"
            "                    file that exists is used)  --lights-timing\n"
            "  --frames N        exit after N processed frames (testing)\n"
            "  --stats S         stderr stats interval in s (default 10, 0 = off)\n"
            "  --verbose\n"
            "config: $ROADSENSE_CONFIG_DIR (default /etc/roadsense)/signd.json, keys backend, det_backend, cls_backend,\n"
            "        models_dir, det_model, cls_model, cls_names, threads, cpus, nice, spin, ncnn_fp16, ncnn_flushall,\n"
            "        det_thr, cls_gate, nms_iou, cls_margin, max_age, min_hits, iou_match, dist_match, dist_match_new, crop_margin, max_crops, max_hz,\n"
            "        appearance, verifier, glow_px, glow_share, glow_alone, verify_thr, verify_vouch, class_min_score (CLI wins)\n"
            "env: ROADSENSE_RUNDIR (default /run/roadsense), ROADSENSE_RING (default /dev/shm/roadsense-frames)\n");
}

// --lights-ru-thr: "person=0.4,car=0.45" (names of road_user_name()) or eight numbers in class order.
bool parse_ru_thr(const std::string &s, float *thr) {
    if (s.find('=') == std::string::npos) {
        float v[8];
        const char *p = s.c_str();
        for (int i = 0; i < 8; ++i) {
            char *e;
            v[i] = strtof(p, &e);
            if (e == p) return false;
            p = *e == ',' ? e + 1 : e;
        }
        std::copy(v, v + 8, thr);
        return true;
    }
    size_t a = 0;
    while (a < s.size()) {
        const size_t b = std::min(s.find(',', a), s.size());
        const std::string kv = s.substr(a, b - a);
        const size_t q = kv.find('=');
        if (q == std::string::npos) return false;
        int c = -1;
        for (int i = 0; i < 8; ++i)
            if (kv.compare(0, q, roadsense::signs::road_user_name(i)) == 0 && q == strlen(roadsense::signs::road_user_name(i))) c = i;
        if (c < 0) return false;
        thr[c] = strtof(kv.c_str() + q + 1, nullptr);
        a = b + 1;
    }
    return true;
}

void load_config(Options &o) {
    std::string s;
    if (!read_file(config_dir() + "/signd.json", &s)) return;
    double v;
    std::string str;
    if (json_string(s, "backend", &str)) o.pc.det_backend = o.pc.cls_backend = str;
    if (json_string(s, "det_backend", &str)) o.pc.det_backend = str;
    if (json_string(s, "cls_backend", &str)) o.pc.cls_backend = str;
    if (json_string(s, "models_dir", &str)) o.pc.models_dir = str;
    if (json_string(s, "det_model", &str)) o.pc.det_model = str;
    if (json_string(s, "cls_model", &str)) o.pc.cls_model = str;
    if (json_string(s, "cls_names", &str)) o.pc.cls_names = str;
    if (json_number(s, "threads", &v)) o.pc.threads = int(v);
    if (json_string(s, "cpus", &str)) o.cpus = str;
    if (json_number(s, "nice", &v)) o.nice = int(v);
    if (json_number(s, "spin", &v)) o.pc.allow_spinning = v != 0;
    if (json_number(s, "ncnn_fp16", &v)) o.pc.ncnn_fp16 = v != 0;
    if (json_number(s, "ncnn_flushall", &v)) o.pc.ncnn_flushall = v != 0;
    if (json_number(s, "det_thr", &v)) o.pc.det_thr = float(v);
    if (json_number(s, "cls_gate", &v)) o.pc.cls_gate = o.tc.cls_gate = float(v);
    if (json_number(s, "nms_iou", &v)) o.pc.nms_iou = float(v);
    if (json_number(s, "cls_margin", &v)) o.pc.cls_margin = float(v);
    if (json_number(s, "max_age", &v)) o.tc.max_age = int(v);
    if (json_number(s, "min_hits", &v)) o.tc.min_hits = int(v);
    if (json_number(s, "iou_match", &v)) o.tc.iou_match = float(v);
    if (json_number(s, "dist_match", &v)) o.tc.dist_match = float(v);
    if (json_number(s, "dist_match_new", &v)) o.tc.dist_match_new = float(v);
    if (json_number(s, "crop_margin", &v)) o.tc.crop_margin = float(v);
    if (json_number(s, "max_crops", &v)) o.max_crops = size_t(v);
    if (json_number(s, "max_hz", &v)) o.max_hz = v;
    if (json_number(s, "appearance", &v)) o.pc.appearance = v != 0;
    if (json_string(s, "verifier", &str)) o.pc.verifier_model = str;
    if (json_number(s, "glow_px", &v)) o.tc.glow_px = float(v);
    if (json_number(s, "glow_share", &v)) o.tc.glow_share = float(v);
    if (json_number(s, "glow_alone", &v)) o.tc.glow_alone = v != 0;
    if (json_number(s, "verify_vouch", &v)) o.tc.verify_vouch = float(v);
    if (json_number(s, "verify_thr", &v)) o.tc.verify_thr = float(v);
    if (json_string(s, "class_min_score", &str) && !parse_class_min_score(str, &o.tc.class_min_score))
        fprintf(stderr, "signd: bad class_min_score in signd.json\n");
    if (json_string(s, "gpu_sync", &str)) o.gpu_sync = str;
    if (json_string(s, "gpu_sched", &str)) o.gpu_sched = str;
    if (json_number(s, "gpu_sync_timeout_ms", &v)) o.gpu_sync_timeout_ms = int(v);
    if (json_number(s, "lights", &v)) o.lights = v != 0;
    if (json_string(s, "lights_backend", &str)) o.lc.backend = str;
    if (json_string(s, "lights_model", &str)) o.lc.model = str;
    if (json_string(s, "lights_state", &str)) o.lc.state = str;
    if (json_string(s, "lights_cpus", &str)) o.lights_cpus = str;
    if (json_number(s, "lights_hz", &v)) o.lights_hz = v;
    if (json_string(s, "lights_idle", &str)) o.lights_idle = str;
    if (json_number(s, "lights_threads", &v)) o.lc.threads = int(v);
    if (json_number(s, "lights_thr", &v)) o.lc.det_thr = float(v);
    if (json_number(s, "lights_road_users", &v)) o.lc.road_users = v != 0;
    if (json_string(s, "lights_ru_thr", &str) && !parse_ru_thr(str, o.lc.ru_thr))
        fprintf(stderr, "signd: bad lights_ru_thr in signd.json\n");
}

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
        if (a == "--backend") opt.pc.det_backend = opt.pc.cls_backend = val();
        else if (a == "--det-backend") opt.pc.det_backend = val();
        else if (a == "--cls-backend") opt.pc.cls_backend = val();
        else if (a == "--models-dir") opt.pc.models_dir = val();
        else if (a == "--ncnn-fp32") opt.pc.ncnn_fp16 = false;
        else if (a == "--ncnn-no-flushall") opt.pc.ncnn_flushall = false;
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
        else if (a == "--dist-match") opt.tc.dist_match = float(atof(val().c_str()));
        else if (a == "--dist-match-new") opt.tc.dist_match_new = float(atof(val().c_str()));
        else if (a == "--crop-margin") opt.tc.crop_margin = float(atof(val().c_str()));
        else if (a == "--roi2") {
            if (!parse_floats(val(), opt.pc.roi2, 4)) return usage(), 2;
        } else if (a == "--roi2-every") opt.pc.roi2_every = std::max(1, atoi(val().c_str()));
        else if (a == "--no-appearance") opt.pc.appearance = false;
        else if (a == "--verifier") opt.pc.verifier_model = val();
        else if (a == "--glow-px") opt.tc.glow_px = float(atof(val().c_str()));
        else if (a == "--glow-share") opt.tc.glow_share = float(atof(val().c_str()));
        else if (a == "--glow-alone") opt.tc.glow_alone = true;
        else if (a == "--verify-vouch") opt.tc.verify_vouch = float(atof(val().c_str()));
        else if (a == "--verify-thr") opt.tc.verify_thr = float(atof(val().c_str()));
        else if (a == "--class-min-score") {
            if (!parse_class_min_score(val(), &opt.tc.class_min_score)) {
                fprintf(stderr, "bad --class-min-score (prefix=score,...)\n");
                return 2;
            }
        }
        else if (a == "--max-crops") opt.max_crops = size_t(atol(val().c_str()));
        else if (a == "--max-hz") opt.max_hz = atof(val().c_str());
        else if (a == "--gpu-sync") opt.gpu_sync = val();
        else if (a == "--gpu-sched") opt.gpu_sched = val();
        else if (a == "--gpu-sync-timeout") opt.gpu_sync_timeout_ms = atoi(val().c_str());
        else if (a == "--timing") opt.timing = true;
        else if (a == "--lights") opt.lights = true;
        else if (a == "--lights-backend") opt.lc.backend = val();
        else if (a == "--lights-model") opt.lc.model = val();
        else if (a == "--lights-in") {
            if (sscanf(val().c_str(), "%dx%d", &opt.lc.in_w, &opt.lc.in_h) != 2) return usage(), 2;
        } else if (a == "--lights-roi") {
            if (!parse_floats(val(), opt.lc.roi, 4)) return usage(), 2;
        } else if (a == "--lights-roi2") {
            if (!parse_floats(val(), opt.lc.roi2, 4)) return usage(), 2;
        } else if (a == "--lights-thr-keep") opt.lc.det_thr_keep = float(atof(val().c_str()));
        else if (a == "--lights-coast") opt.ltc.report_coast = atoi(val().c_str());
        else if (a == "--lights-hz") opt.lights_hz = atof(val().c_str());
        else if (a == "--lights-idle") opt.lights_idle = val();
        else if (a == "--lights-threads") opt.lc.threads = atoi(val().c_str());
        else if (a == "--lights-cpus") opt.lights_cpus = val();
        else if (a == "--lights-thr") opt.lc.det_thr = float(atof(val().c_str()));
        else if (a == "--lights-state") opt.lc.state = val();
        else if (a == "--lights-timing") opt.lights_timing = true;
        else if (a == "--lights-signs") opt.lights_signs = true, opt.lc.sign_cls = 9;
        else if (a == "--lights-road-users") opt.lc.road_users = true;
        else if (a == "--lights-ru-roi") {
            if (!parse_floats(val(), opt.lc.ru_roi, 4)) return usage(), 2;
        } else if (a == "--lights-ru-hz") opt.lc.ru_hz = atof(val().c_str());
        else if (a == "--lights-ru-thr") {
            if (!parse_ru_thr(val(), opt.lc.ru_thr)) return usage(), 2;
        }
        else if (a == "--lights-signs-thr") opt.lc.sign_thr = float(atof(val().c_str()));
        else if (a == "--lights-signs-keep") opt.lights_signs = true, opt.lc.sign_cls = 9, opt.tc.external_starts = false;
        else if (a == "--lights-signs-nohits") opt.tc.external_hits = false;
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
    if (opt.gpu_sync != "none" && opt.gpu_sync != "after" && opt.gpu_sync != "turn") {
        fprintf(stderr, "signd: bad --gpu-sync '%s' (none, after, turn)\n", opt.gpu_sync.c_str());
        return 2;
    }

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
    pthread_setname_np(pthread_self(), "sg-signs");  // thread names: per-stage CPU accounting; pools inherit

    std::unique_ptr<SignPipeline> pipe;
    try {
        pipe = std::make_unique<SignPipeline>(opt.pc);
    } catch (const std::exception &e) {
        fprintf(stderr, "signd: model load failed: %s\n", e.what());
        return 1;
    }
    fprintf(stderr,
            "signd: round red checks: glow %s (px %.2f share %.2f%s), verifier %s (thr %.2f vouch %.2f), %zu class min scores\n",
            opt.pc.appearance ? "on" : "off", opt.tc.glow_px, opt.tc.glow_share, opt.tc.glow_alone ? ", alone" : "",
            pipe->has_verifier() ? opt.pc.verifier_model.c_str() : "none", opt.tc.verify_thr, opt.tc.verify_vouch,
            opt.tc.class_min_score.size());
    if (pipe->has_view2())
        fprintf(stderr, "signd: second view roi %.3f,%.3f,%.3f,%.3f on every %d. frame\n", opt.pc.roi2[0], opt.pc.roi2[1],
                opt.pc.roi2[2], opt.pc.roi2[3], opt.pc.roi2_every);
    fprintf(stderr,
            "signd: detector %dx%d on %s (%s), classifier on %s (%s), %zu classes (%zu merged), %d threads on cpus %s, "
            "det_thr %.2f cls_gate %.2f nms %.2f\n",
            pipe->input_w(), pipe->input_h(), pipe->det_backend(), pipe->config().det_model.c_str(), pipe->cls_backend(),
            pipe->config().cls_model.c_str(), pipe->names().raw.size(), pipe->names().merged.size(), opt.pc.threads,
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

    // GPU schedule (optional): align the detector with the end of a modeld run so it fills modeld's
    // CPU gap instead of stalling a modeld run halfway. Never blocks if modeld is not there.
    GpuSched gs;
    // With lights on the GPU in turn mode both detectors take turns from modeld through `gs`; the mutex makes
    // sure only one of them asks at a time (the shm has one want/turn pair).
    std::mutex gpu_mu;
    const bool lights_gpu_turn = opt.lights && opt.lc.backend == "ncnn" && opt.gpu_sync == "turn";
    bool mu_held = false;
    bool turn_held = false;
    Clock::time_point turn_t0;
    uint64_t stats_ticks = 0, stats_turns = 0;
    double stats_wait = 0;
    if (opt.gpu_sync != "none") {
        const std::string gp = opt.gpu_sched.empty() ? rundir + "/gpu.sched" : opt.gpu_sched;
        gs.open(gp, false);  // may not exist yet: peer_alive() retries once per second
        const bool want_turn = opt.gpu_sync == "turn";
        pipe->before_det = [&gs, &turn_held, &turn_t0, &stats_ticks, &stats_turns, &opt, want_turn, &gpu_mu, &mu_held,
                            lights_gpu_turn] {
            if (lights_gpu_turn) gpu_mu.lock(), mu_held = true;
            bool tick = false;
            turn_held = gs.wait_run_end(opt.gpu_sync_timeout_ms, want_turn, &tick);
            turn_t0 = Clock::now();
            stats_ticks += tick;
            stats_turns += turn_held;
        };
        pipe->after_det = [&gs, &turn_held, &turn_t0, &gpu_mu, &mu_held] {
            if (turn_held) gs.end_turn(std::chrono::duration<double, std::milli>(Clock::now() - turn_t0).count());
            turn_held = false;
            if (mu_held) mu_held = false, gpu_mu.unlock();
        };
        fprintf(stderr, "signd: GPU sync %s via %s (timeout %d ms)\n", opt.gpu_sync.c_str(), gp.c_str(),
                opt.gpu_sync_timeout_ms);
    }

    std::mutex prop_mu;  // --lights-signs: light thread -> sign thread
    std::vector<SignProposals> proposals;

    // ---- traffic lights: own thread, own ring reader, own socket
    if (opt.lc.models_dir.empty() || opt.lc.models_dir == LightsConfig().models_dir) opt.lc.models_dir = opt.pc.models_dir;
    auto lights_main = [&]() {
        pthread_setname_np(pthread_self(), "sg-lights");
        const std::string lcpus = opt.lights_cpus.empty() ? opt.cpus : opt.lights_cpus;
        if (!lcpus.empty() && lcpus != "none") {
            cpu_set_t set;
            if (parse_cpus(lcpus, &set)) pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        }
        std::unique_ptr<LightPipeline> lp;
        try {
            lp = std::make_unique<LightPipeline>(opt.lc);
        } catch (const std::exception &e) {
            fprintf(stderr, "signd: lights disabled: %s\n", e.what());
            return;
        }
        if (opt.lc.det_thr_keep > 0) opt.ltc.new_score = opt.lc.det_thr;
        if (lp->has_view2())
            fprintf(stderr, "signd: lights: second view roi %.3f,%.3f,%.3f,%.3f on every other frame\n", opt.lc.roi2[0],
                    opt.lc.roi2[1], opt.lc.roi2[2], opt.lc.roi2[3]);
        if (opt.lc.road_users)
            fprintf(stderr, "signd: lights: road users on (thr %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f; governor key road_users_hz)\n",
                    opt.lc.ru_thr[0], opt.lc.ru_thr[1], opt.lc.ru_thr[2], opt.lc.ru_thr[3], opt.lc.ru_thr[4], opt.lc.ru_thr[5],
                    opt.lc.ru_thr[6], opt.lc.ru_thr[7]);
        fprintf(stderr, "signd: lights: thr %.2f keep %.2f, coast %d frames\n", opt.lc.det_thr, opt.lc.det_thr_keep,
                opt.ltc.report_coast);
        fprintf(stderr, "signd: lights: %s on %s, input %dx%d, roi %.2f,%.2f,%.2f,%.2f, %.1f Hz, state %s, cpus %s%s\n",
                lp->config().model.c_str(), lp->backend(), opt.lc.in_w, opt.lc.in_h, opt.lc.roi[0], opt.lc.roi[1],
                opt.lc.roi[2], opt.lc.roi[3], opt.lights_hz, opt.lc.state.c_str(), lcpus.c_str(),
                lights_gpu_turn ? ", GPU turns" : "");
        bool l_mu = false, l_turn = false;
        uint64_t l_turns = 0;
        Clock::time_point l_t0;
        if (lights_gpu_turn) {
            lp->before_det = [&] {
                gpu_mu.lock();
                l_mu = true;
                bool tick = false;
                l_turn = gs.wait_run_end(opt.gpu_sync_timeout_ms, true, &tick);
                l_turns += l_turn;
                l_t0 = Clock::now();
            };
            // end_turn(0): the shared det_ms (modeld's lead) stays the sign detector's; for the shorter light
            // detector modeld simply waits for the turn to come back.
            lp->after_det = [&] {
                if (l_turn) gs.end_turn(0);
                l_turn = false;
                if (l_mu) l_mu = false, gpu_mu.unlock();
            };
        }
        JsonLineServer lsrv;
        std::string lerr;
        if (!lsrv.listen(rundir + "/lightd.sock", &lerr)) {
            fprintf(stderr, "signd: lights: %s\n", lerr.c_str());
            return;
        }
        LightTracker lt(opt.ltc);
        RingReader lr(ring_path());
        FrameView lf;
        std::vector<LightDet> ld;
        std::vector<LightTrackReport> lrep;
        uint64_t l_last = 0, l_n = 0, l_frames = 0;
        double l_ms = 0, l_det = 0, l_wait = 0;
        auto l_stats_t0 = Clock::now(), l_next = Clock::now(), u_next = Clock::now(), ru_next = Clock::now();
        double ru_ms = 0;
        uint64_t ru_n = 0;
        // --lights-hz, lowered by the load governor's `lights_hz` cap under heat (roadsense camerad/governor.h);
        // the tracker's ages count light frames, so they stretch in time with the rate.
        roadsense::GovernorReader l_gov;
        roadsense::GovernorReader l_demand(rundir + "/demand");
        uint64_t l_idle = 0, l_idle_at = 0;  // frame periods without a pass since the last stats line; pattern index
        while (g_run) {
            if (!lr.is_open() && !lr.open()) {
                lsrv.service();
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
            const auto now = Clock::now();
            if (now < l_next) {
                lsrv.service();
                std::this_thread::sleep_for(std::min(l_next - now, Clock::duration(std::chrono::milliseconds(20))));
                continue;
            }
            const auto l_period = std::chrono::microseconds(int64_t(1e6 / std::max(0.5, l_gov.limit("lights_hz", opt.lights_hz))));
            int view = lp->has_view2() ? int(l_frames & 1) : 0;   // wide, zoom, wide, ...
            if (!opt.lights_idle.empty() && lt.live() == 0 && l_demand.cap("lights_idle") != 0) {
                // nothing tracked: the idle pattern decides whether this period gets a pass, and of which view
                const char c = opt.lights_idle[l_idle_at++ % opt.lights_idle.size()];
                if (c == '.') {  // no pass and no message: consumers hold the last one, which lists no light
                    ++l_idle;
                    l_next = Clock::now() + l_period;
                    lsrv.service();
                    continue;
                }
                view = c == 'Z' && lp->has_view2() ? 1 : 0;
            } else {
                l_idle_at = 0;
            }
            if (!lr.next(lf, 200)) {
                lsrv.service();
                continue;
            }
            l_next = Clock::now() + l_period;
            if (lf.seq < l_last) lt.reset();
            l_last = lf.seq;
            if (lf.format != kFormatNV12 || (lf.width & 1) || (lf.height & 1)) continue;
            Nv12View v;
            v.y = lf.y, v.uv = lf.uv, v.width = int(lf.width), v.height = int(lf.height);
            v.y_stride = int(lf.y_stride), v.uv_stride = int(lf.uv_stride);
            LightsTiming t;
            l_frames = uint64_t(view) + 1;  // the alternation goes on from this view
            // road users (non-critical): paused by the governor's `road_users_hz` cap (0), thinned by a positive one
            int users_state = 0;
            if (opt.lc.road_users) {
                const double ucap = l_gov.cap("road_users_hz");
                const auto un = Clock::now();
                const bool due = ucap < 0 || (ucap > 0 && un >= u_next);
                if (due && ucap > 0) u_next = un + std::chrono::microseconds(std::max<int64_t>(0, int64_t(1e6 / ucap) - 20000));
                lp->set_road_users(due);
                users_state = due ? 1 : (ucap == 0 ? 2 : 0);
            }
            try {
                lp->process(v, ld, &t, view);
            } catch (const std::exception &e) {
                if (l_turn) gs.end_turn(0), l_turn = false;
                if (l_mu) l_mu = false, gpu_mu.unlock();
                fprintf(stderr, "signd: lights failed: %s\n", e.what());
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }
            const auto tt0 = Clock::now();
            const float *vr = lp->view_roi(view);
            const float seen[4] = {vr[0] * v.width, vr[1] * v.height, vr[2] * v.width, vr[3] * v.height};
            lt.update(lf.seq, lf.ts_mono_ns, ld, lrep, seen);
            const double ms = t.total_ms + std::chrono::duration<double, std::milli>(Clock::now() - tt0).count();
            lsrv.publish(lights_json(lf.seq, lf.ts_mono_ns, lf.ts_real_ns, ms, v.width, v.height, lp->config(), lrep,
                                     opt.lights_timing ? &ld : nullptr, opt.lights_timing ? &t : nullptr, view,
                                     &lp->road_users(), users_state));
            if (opt.lights_signs) {  // sign proposals of this light frame -> sign classifier -> sign thread
                SignProposals sp;
                sp.seq = lf.seq, sp.ts_mono_ns = lf.ts_mono_ns, sp.ts_real_ns = lf.ts_real_ns;
                std::copy(seen, seen + 4, sp.seen);
                for (const LightDet &b : lp->sign_boxes()) {
                    Detection d;
                    d.x0 = b.x0, d.y0 = b.y0, d.x1 = b.x1, d.y1 = b.y1, d.score = b.score;
                    d.external = true;
                    sp.dets.push_back(std::move(d));
                }
                if (!sp.dets.empty()) pipe->classify(v, sp.dets);
                std::lock_guard<std::mutex> lk(prop_mu);
                if (proposals.size() < 8) proposals.push_back(std::move(sp));
            }
            if (lp->has_ru_view() && users_state != 2) {  // the road-user view: after the light frame, never before it
                const double ucap = l_gov.cap("road_users_hz");
                const double hz = ucap > 0 ? std::min(ucap, opt.lc.ru_hz) : opt.lc.ru_hz;
                const auto un = Clock::now();
                if (ucap != 0 && hz > 0 && un >= ru_next) {
                    ru_next = un + std::chrono::microseconds(int64_t(1e6 / hz));
                    std::vector<LightDet> none;
                    LightsTiming t3;
                    try {
                        lp->process(v, none, &t3, 2);
                        lsrv.publish(road_users_json(lf.seq, lf.ts_mono_ns, lf.ts_real_ns, t3.total_ms, v.width, v.height,
                                                     lp->view_roi(2), lp->road_users()));
                        ru_ms += t3.total_ms, ++ru_n;
                    } catch (const std::exception &e) {
                        fprintf(stderr, "signd: road-user view failed: %s\n", e.what());
                    }
                }
            }
            ++l_n, l_ms += ms, l_det += t.det_ms, l_wait += t.wait_ms;
            const double el = std::chrono::duration<double>(Clock::now() - l_stats_t0).count();
            if (opt.stats_s > 0 && el >= opt.stats_s) {
                fprintf(stderr, "signd: lights %.2f fps, lights_ms %.1f (det %.1f), gpu wait %.1f ms, turns %llu, live %zu, clients %zu\n",
                        l_n / el, l_ms / std::max<uint64_t>(1, l_n), l_det / std::max<uint64_t>(1, l_n),
                        l_wait / std::max<uint64_t>(1, l_n), (unsigned long long)l_turns, lt.live(), lsrv.client_count());
                if (ru_n)
                    fprintf(stderr, "signd: road-user view %.2f runs/s, %.1f ms\n", ru_n / el, ru_ms / double(ru_n));
                if (l_idle) fprintf(stderr, "signd: lights idle: %llu frame periods without a pass, %llu passes\n",
                                    (unsigned long long)l_idle, (unsigned long long)l_n);
                l_idle = 0;
                l_stats_t0 = Clock::now();
                l_n = 0, l_ms = l_det = l_wait = 0, l_turns = 0, ru_ms = 0, ru_n = 0;
            }
        }
        lsrv.close();
    };
    std::thread lights_thread;
    if (opt.lights) lights_thread = std::thread(lights_main);

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
    auto next_allowed = Clock::now();
    int last_view = -1;
    uint64_t last_pub_seq = 0, stats_late = 0;  // --lights-signs: newest published seq, light updates too late
    std::vector<EndedTrack> carry;               // ended tracks of light updates older than the last message

    auto publish = [&](const FrameMeta &m, double ms, const Timing *t) {
        crop_paths.clear();
        for (const EndedTrack &e : ended) crop_paths.push_back(crops.save(e.id, e.crop));
        ExtraFields ex;
        if (opt.timing) ex.timing = t;
        ex.view = last_view;
        srv.publish(signs_json(m, ms, reports, ended, crop_paths, ex));
    };

    roadsense::GovernorReader gov;
    bool next_allowed_set = false;
    while (g_run) {
        if (!rr.is_open() && !rr.open()) {
            if (!waiting_logged) fprintf(stderr, "signd: waiting for ring %s\n", ring_path().c_str());
            waiting_logged = true;
            srv.service();
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }
        // duty cycle: wait out the period, then take the newest frame. --max-hz (0 = as fast as the
        // pipeline goes), lowered by the load governor's `signs_hz` cap under heat.
        const double hz = gov.limit("signs_hz", opt.max_hz);
        if (hz > 0 || next_allowed_set) {
            const auto now = Clock::now();
            if (now < next_allowed) {
                srv.service();
                std::this_thread::sleep_for(std::min(next_allowed - now, std::chrono::steady_clock::duration(std::chrono::milliseconds(20))));
                continue;
            }
        }
        if (!rr.next(fv, 200)) {
            srv.service();
            continue;
        }
        next_allowed_set = hz > 0;
        if (hz > 0) next_allowed = Clock::now() + std::chrono::microseconds(int64_t(1e6 / hz));
        if (fv.seq < last_seq) {  // camerad restarted: the old tracks cannot continue
            if (opt.lights_signs) {
                std::lock_guard<std::mutex> lk(prop_mu);
                proposals.clear();
            }
            last_pub_seq = 0;
            carry.clear();
            reports.clear();
            ended.clear();
            tracker.flush(ended);
            if (!ended.empty()) publish(last_meta, 0.0, nullptr);
        } else if (last_seq) {
            skipped += fv.seq - last_seq - 1;
        }
        last_seq = fv.seq;
        if (opt.lights_signs) {  // light frames older than this frame: tracker updates of their own, in order
            std::vector<SignProposals> todo;
            {
                std::lock_guard<std::mutex> lk(prop_mu);
                for (auto it = proposals.begin(); it != proposals.end();)
                    if (it->seq < fv.seq) todo.push_back(std::move(*it)), it = proposals.erase(it);
                    else ++it;
            }
            for (SignProposals &sp : todo) {
                Nv12View cv;  // external boxes never crop: the tracker does not read the frame for them
                cv.y = fv.y, cv.uv = fv.uv, cv.width = int(fv.width), cv.height = int(fv.height);
                cv.y_stride = int(fv.y_stride), cv.uv_stride = int(fv.uv_stride);
                ended.clear();
                tracker.update(sp.seq, cv, sp.dets, reports, ended, sp.seen);
                if (sp.seq > last_pub_seq) {  // in order: a `signs` message of its own (view 2 = a light frame)
                    FrameMeta pm;
                    pm.seq = sp.seq, pm.ts_mono_ns = sp.ts_mono_ns, pm.ts_real_ns = sp.ts_real_ns;
                    last_view = 2;
                    if (!carry.empty()) ended.insert(ended.begin(), std::make_move_iterator(carry.begin()),
                                                     std::make_move_iterator(carry.end())), carry.clear();
                    publish(pm, 0.0, nullptr);
                    last_pub_seq = sp.seq;
                } else {  // older than the last message: no message (seq stays monotonic); its ended tracks go
                          // out with the next one
                    for (EndedTrack &e : ended) carry.push_back(std::move(e));
                    ++stats_late;
                }
            }
        }
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
        const int view = pipe->view_for(processed);  // --roi2: second view on every roi2_every-th frame
        try {
            pipe->process(v, dets, &t, view);
        } catch (const std::exception &e) {
            if (turn_held) gs.end_turn(), turn_held = false;
            if (mu_held) mu_held = false, gpu_mu.unlock();
            fprintf(stderr, "signd: inference failed: %s\n", e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        ended.clear();
        const auto tt0 = Clock::now();
        float seen[4];
        pipe->view_rect(view, v.width, v.height, seen);
        tracker.update(fv.seq, v, dets, reports, ended, pipe->has_view2() ? seen : nullptr);
        last_view = pipe->has_view2() ? view : -1;
        const double detect_ms = t.total_ms + std::chrono::duration<double, std::milli>(Clock::now() - tt0).count();
        FrameMeta m;
        m.seq = fv.seq;
        m.ts_mono_ns = fv.ts_mono_ns;
        m.ts_real_ns = fv.ts_real_ns;
        last_meta = m;
        if (!carry.empty()) ended.insert(ended.begin(), std::make_move_iterator(carry.begin()),
                                         std::make_move_iterator(carry.end())), carry.clear();
        publish(m, detect_ms, &t);
        last_pub_seq = std::max(last_pub_seq, fv.seq);
        ++processed;
        ++stats_n;
        stats_ms += detect_ms;
        stats_det += t.det_ms;
        stats_wait += t.wait_ms;
        if (opt.verbose)
            fprintf(stderr, "signd: seq %llu: %zu raw, %zu reported, %zu ended, %.1f ms (pre %.1f det %.1f cls %.1f)\n",
                    (unsigned long long)fv.seq, dets.size(), reports.size(), ended.size(), detect_ms, t.pre_ms,
                    t.det_ms, t.cls_ms);
        const double el = std::chrono::duration<double>(Clock::now() - stats_t0).count();
        if (opt.stats_s > 0 && el >= opt.stats_s) {
            fprintf(stderr,
                    "signd: %.2f fps, detect_ms %.1f (det %.1f), live tracks %zu, frames skipped %llu, clients %zu, "
                    "gpu wait %.1f ms, ticks %llu, turns %llu\n",
                    stats_n / el, stats_ms / stats_n, stats_det / stats_n, tracker.live(),
                    (unsigned long long)skipped, srv.client_count(), stats_wait / stats_n,
                    (unsigned long long)stats_ticks, (unsigned long long)stats_turns);
            if (opt.lights_signs) fprintf(stderr, "signd: lights-signs: %llu light updates older than the last message\n",
                                          (unsigned long long)stats_late);
            stats_t0 = Clock::now();
            stats_n = 0;
            stats_ms = stats_det = stats_wait = 0;
            stats_ticks = stats_turns = 0;
        }
        if (opt.max_frames && processed >= opt.max_frames) break;
    }
    // Shutdown: end the live tracks so their crops and ends are published.
    reports.clear();
    ended.clear();
    tracker.flush(ended);
    if (!ended.empty()) publish(last_meta, 0.0, nullptr);
    srv.close();
    g_run = false;
    if (lights_thread.joinable()) lights_thread.join();
    fprintf(stderr, "signd: exit after %llu frames\n", (unsigned long long)processed);
    return 0;
}
