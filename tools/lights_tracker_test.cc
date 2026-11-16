// lights_tracker_test: checks of LightTracker (signd/lights.h) that need no model and no frame:
// coasting reports, the keep threshold, and the two alternating views. Exit code 0 = all passed.
//   cmake --build build --target lights_tracker_test && build/lights_tracker_test
#include <cstdio>
#include <vector>

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

LightDet det(float cx, float cy, float h, float score, int state = kRed) {
    LightDet d;
    d.x0 = cx - 0.2f * h, d.x1 = cx + 0.2f * h, d.y0 = cy - 0.5f * h, d.y1 = cy + 0.5f * h;
    d.score = score, d.state = state, d.state_conf = 0.95f;
    return d;
}

struct Run {
    LightTracker trk;
    std::vector<LightTrackReport> out;
    uint64_t seq = 0;
    explicit Run(const LightTrackerConfig &c) : trk(c) {}
    const std::vector<LightTrackReport> &step(const std::vector<LightDet> &d, const float *seen = nullptr) {
        ++seq;
        trk.update(seq, int64_t(seq) * 333000000, d, out, seen);
        return out;
    }
};

void test_matched_only_by_default() {
    Run r{LightTrackerConfig()};
    CHECK(r.step({det(600, 200, 30, 0.6f)}).empty());  // one hit: not confirmed
    CHECK(r.step({det(602, 200, 31, 0.6f)}).size() == 1);
    CHECK(r.out[0].coast == 0);
    CHECK(r.step({}).empty());  // a missed frame: the light is gone from the message (old behaviour)
    CHECK(r.step({det(606, 200, 33, 0.6f)}).size() == 1);
    CHECK(r.out[0].hits == 3);
}

void test_coasting_reports() {
    LightTrackerConfig c;
    c.report_coast = 2;
    Run r{c};
    r.step({det(600, 200, 30, 0.6f)});
    r.step({det(604, 200, 31, 0.6f)});
    const int id = r.out[0].id;
    CHECK(r.step({}).size() == 1);  // missed once: still listed
    CHECK(r.out[0].id == id && r.out[0].coast == 1 && r.out[0].state == kRed && r.out[0].frame_state == kUnknown);
    CHECK(r.out[0].box[0] > 604 - 0.2f * 31);  // moved on with its velocity
    CHECK(r.step({}).size() == 1 && r.out[0].coast == 2);
    CHECK(r.step({}).empty());  // third miss: not listed any more
    CHECK(r.step({det(620, 200, 34, 0.6f)}).size() == 1);  // the track itself lives on (max_age 6)
    CHECK(r.out[0].id == id && r.out[0].coast == 0);
    // a track that was never confirmed is not reported while coasting
    Run q{c};
    q.step({det(100, 100, 20, 0.6f)});
    CHECK(q.step({}).empty());
}

void test_keep_threshold() {
    LightTrackerConfig c;
    c.new_score = 0.25f;
    Run r{c};
    CHECK(r.step({det(600, 200, 30, 0.15f)}).empty());
    CHECK(r.step({det(600, 200, 30, 0.15f)}).empty());  // weak boxes never start a track
    CHECK(r.trk.live() == 0);
    r.step({det(600, 200, 30, 0.5f)});
    CHECK(r.step({det(603, 200, 31, 0.15f)}).size() == 1);  // but they continue one
    CHECK(r.out[0].hits == 2);
    // ... only close to it: a weak box 1.2 box sizes away is not this light (a strong one would be)
    CHECK(r.step({det(603 + 1.2f * 31, 200, 31, 0.15f)}).empty());
    CHECK(r.trk.live() == 1);
}

void test_two_views() {
    LightTrackerConfig c;
    c.report_coast = 1;
    c.max_age = 2;
    Run r{c};
    const float wide[4] = {0, 0, 1280, 446}, zoom[4] = {160, 90, 1120, 410};
    // a near light at the top left, outside the zoom view: seen in the wide frames only
    for (int i = 0; i < 12; ++i) {
        const bool w = i % 2 == 0;
        r.step(w ? std::vector<LightDet>{det(80, 40, 60, 0.7f)} : std::vector<LightDet>{}, w ? wide : zoom);
        if (i >= 2) {
            CHECK(r.out.size() == 1);  // listed in every frame: matched in the wide ones, coasting in between
            if (r.out.size() == 1) CHECK(r.out[0].coast == (w ? 0 : 1));
        }
    }
    CHECK(r.trk.live() == 1);  // the zoom frames did not age it (max_age 2 would have ended it otherwise)
    // once it is really gone it ends: missed in the wide frames where it could be seen
    for (int i = 0; i < 8; ++i) r.step({}, i % 2 == 0 ? wide : zoom);
    CHECK(r.trk.live() == 0);
    // a track that left every view does not live for ever
    Run q{c};
    q.step({det(80, 40, 60, 0.7f)}, wide);
    q.step({det(80, 40, 60, 0.7f)}, wide);
    for (int i = 0; i < 10; ++i) q.step({}, zoom);
    CHECK(q.trk.live() == 0);
}
}  // namespace

int main() {
    test_matched_only_by_default();
    test_coasting_reports();
    test_keep_threshold();
    test_two_views();
    if (failures) return fprintf(stderr, "lights_tracker_test: %d check(s) failed\n", failures), 1;
    printf("lights_tracker_test: ok\n");
    return 0;
}
