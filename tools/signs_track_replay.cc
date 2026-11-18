// signs_track_replay: signd's sign tracker (signd/tracker.h, the same code and options as signd and
// signd_replay) on detections computed beforehand (tools/signs_eval/cache_dets.py, binary .dets files),
// for evaluating rates, view schedules and extra proposal sources without re-running the detector
// (roadsense lane L24-S). One `signs` JSON line per scheduled frame, with "frame" (the ride frame index) added.
//
//   signs_track_replay --dets A=ride.A.dets [--dets B=ride.B.dets --rect B=640,144,1280,528] --schedule FILE
//                      [tracker options as signd_replay] > signs.jsonl
//   schedule: one line per processed frame, "FRAME NAME" or "FRAME NAME+NAME2" (detections of both sources,
//   merged by NMS); a source with --rect is a window: the tracker does not age tracks outside it on its frames.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "output.h"
#include "tracker.h"
#include "yolo.h"

using namespace roadsense::signs;

namespace {

struct Source {
    int nc = 0;
    std::map<int, std::vector<Detection>> by_frame;
    bool has_rect = false;
    float rect[4] = {0, 0, 0, 0};
};

bool load_dets(const std::string &path, Source &s, float det_thr) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    int32_t ver, nc, nframes;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "L24D", 4) || fread(&ver, 4, 1, f) != 1 || fread(&nc, 4, 1, f) != 1 ||
        fread(&nframes, 4, 1, f) != 1) {
        fclose(f);
        return false;
    }
    s.nc = nc;
    for (int i = 0; i < nframes; ++i) {
        int32_t fr, n;
        if (fread(&fr, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1) break;
        auto &v = s.by_frame[fr];
        for (int k = 0; k < n; ++k) {
            float b[8];
            int32_t cls;
            Detection d;
            d.probs.resize(size_t(nc));
            if (fread(b, 4, 8, f) != 8 || fread(&cls, 4, 1, f) != 1 || fread(d.probs.data(), 4, size_t(nc), f) != size_t(nc)) {
                fclose(f);
                return false;
            }
            d.x0 = b[0], d.y0 = b[1], d.x1 = b[2], d.y1 = b[3], d.score = b[4], d.cls_prob = b[5], d.glow = b[6], d.verify = b[7];
            d.cls_raw = cls;
            if (d.score > det_thr) v.push_back(std::move(d));
        }
    }
    fclose(f);
    return true;
}

float iou(const Detection &a, const Detection &b) {
    const float ix = std::max(0.f, std::min(a.x1, b.x1) - std::max(a.x0, b.x0));
    const float iy = std::max(0.f, std::min(a.y1, b.y1) - std::max(a.y0, b.y0));
    const float in = ix * iy;
    return in / ((a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - in + 1e-9f);
}

void usage() {
    fprintf(stderr,
            "usage: signs_track_replay --dets NAME=FILE [--dets ...] [--rect NAME=x0,y0,x1,y1] --schedule FILE --cls-names PATH\n"
            "  [--det-thr X (0.35)] [--cls-gate X] [--max-age N] [--min-hits N] [--iou-match X] [--dist-match X]\n"
            "  [--dist-match-new X] [--glow-px X] [--glow-share X] [--glow-alone] [--verify-thr X] [--verify-vouch X]\n"
            "  [--class-min-score LIST] [--frame-w W (1280)] [--frame-h H (720)]\n");
}

}  // namespace

int main(int argc, char **argv) {
    TrackerConfig tc;
    float det_thr = 0.35f;
    std::string schedule, names_path;
    std::map<std::string, std::string> dets_files;
    std::map<std::string, std::vector<float>> rects;
    int fw = 1280, fh = 720;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) usage(), exit(2);
            return argv[++i];
        };
        auto kv = [&](const std::string &s, std::string *k, std::string *v) {
            const size_t e = s.find('=');
            if (e == std::string::npos) usage(), exit(2);
            *k = s.substr(0, e), *v = s.substr(e + 1);
        };
        std::string k, v;
        if (a == "--dets") kv(val(), &k, &v), dets_files[k] = v;
        else if (a == "--rect") {
            kv(val(), &k, &v);
            std::vector<float> r(4);
            if (sscanf(v.c_str(), "%f,%f,%f,%f", &r[0], &r[1], &r[2], &r[3]) != 4) return usage(), 2;
            rects[k] = r;
        } else if (a == "--schedule") schedule = val();
        else if (a == "--cls-names") names_path = val();
        else if (a == "--det-thr") det_thr = float(atof(val().c_str()));
        else if (a == "--cls-gate") tc.cls_gate = float(atof(val().c_str()));
        else if (a == "--max-age") tc.max_age = atoi(val().c_str());
        else if (a == "--min-hits") tc.min_hits = atoi(val().c_str());
        else if (a == "--iou-match") tc.iou_match = float(atof(val().c_str()));
        else if (a == "--dist-match") tc.dist_match = float(atof(val().c_str()));
        else if (a == "--dist-match-new") tc.dist_match_new = float(atof(val().c_str()));
        else if (a == "--glow-px") tc.glow_px = float(atof(val().c_str()));
        else if (a == "--glow-share") tc.glow_share = float(atof(val().c_str()));
        else if (a == "--glow-alone") tc.glow_alone = true;
        else if (a == "--verify-thr") tc.verify_thr = float(atof(val().c_str()));
        else if (a == "--verify-vouch") tc.verify_vouch = float(atof(val().c_str()));
        else if (a == "--class-min-score") {
            if (!parse_class_min_score(val(), &tc.class_min_score)) return usage(), 2;
        } else if (a == "--frame-w") fw = atoi(val().c_str());
        else if (a == "--frame-h") fh = atoi(val().c_str());
        else return usage(), a == "--help" ? 0 : 2;
    }
    if (dets_files.empty() || schedule.empty() || names_path.empty()) return usage(), 2;
    ClassNames names;
    std::string err;
    if (!names.load(names_path, &err)) return fprintf(stderr, "%s\n", err.c_str()), 1;
    std::map<std::string, Source> src;
    for (const auto &p : dets_files) {
        Source &s = src[p.first];
        if (!load_dets(p.second, s, det_thr)) return fprintf(stderr, "cannot read %s\n", p.second.c_str()), 1;
        if (s.nc != int(names.raw.size())) return fprintf(stderr, "%s: %d classes, names %zu\n", p.second.c_str(), s.nc, names.raw.size()), 1;
        auto r = rects.find(p.first);
        if (r != rects.end()) s.has_rect = true, std::copy(r->second.begin(), r->second.end(), s.rect);
    }
    const bool windows = !rects.empty();
    Tracker tracker(tc, names);
    Nv12Image dummy;
    dummy.width = fw, dummy.height = fh;  // grey frame of the ride's size: the tracker's best crops need one
    dummy.data.assign(size_t(fw) * fh * 3 / 2, 128);
    const Nv12View dv = dummy.view();
    std::vector<TrackReport> reports;
    std::vector<EndedTrack> ended;
    std::vector<std::string> crops;
    std::ifstream sf(schedule);
    std::string line;
    uint64_t seq = 0;
    FrameMeta last;
    int last_frame = 0;
    auto emit = [&](const FrameMeta &m, int frame) {
        crops.assign(ended.size(), std::string());
        std::string j = signs_json(m, 0.0, reports, ended, crops);
        j.pop_back();
        printf("%s,\"frame\":%d}\n", j.c_str(), frame);
    };
    while (std::getline(sf, line)) {
        std::istringstream is(line);
        int frame;
        std::string spec;
        if (!(is >> frame >> spec)) continue;
        std::vector<Detection> dets;
        float seen[4] = {0, 0, float(fw), float(fh)};
        bool full = false;
        int nsrc = 0;
        size_t a = 0;
        while (a <= spec.size()) {
            const size_t b = std::min(spec.find('+', a), spec.size());
            const std::string nm = spec.substr(a, b - a);
            a = b + 1;
            auto it = src.find(nm);
            if (it == src.end()) return fprintf(stderr, "unknown source %s\n", nm.c_str()), 1;
            ++nsrc;
            if (!it->second.has_rect) full = true;
            else std::copy(it->second.rect, it->second.rect + 4, seen);
            auto fit = it->second.by_frame.find(frame);
            if (fit != it->second.by_frame.end()) {
                const size_t n0 = dets.size();
                dets.insert(dets.end(), fit->second.begin(), fit->second.end());
                if (nm[0] == 'L')  // light-detector sources (LW, LZ): external boxes, as signd --lights-signs
                    for (size_t q = n0; q < dets.size(); ++q) dets[q].external = true;
            }
        }
        if (nsrc > 1) {  // merged sources: NMS by score
            std::stable_sort(dets.begin(), dets.end(), [](const Detection &x, const Detection &y) { return x.score > y.score; });
            std::vector<Detection> kept;
            for (Detection &d : dets) {
                bool ok = true;
                for (const Detection &k : kept)
                    if (iou(d, k) > 0.45f) ok = false;
                if (ok) kept.push_back(std::move(d));
            }
            dets.swap(kept);
        } else {
            std::stable_sort(dets.begin(), dets.end(), [](const Detection &x, const Detection &y) { return x.score > y.score; });
        }
        ++seq;
        ended.clear();
        tracker.update(seq, dv, dets, reports, ended, windows && !full ? seen : nullptr);
        FrameMeta m;
        m.seq = seq;
        last = m;
        last_frame = frame;
        emit(m, frame);
    }
    reports.clear();
    ended.clear();
    tracker.flush(ended);
    if (!ended.empty()) emit(last, last_frame);
    return 0;
}
