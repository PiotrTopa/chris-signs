// signs_tracker_test: checks of the sign Tracker (signd/tracker.h) that need no model: the `seen` window of the
// second detector view (roadsense lane L24-S) and that the default path (no window) is unchanged.
// Exit code 0 = all passed. Also run by `ctest`.
#include <cstdio>
#include <fstream>
#include <vector>

#include "tracker.h"

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

Detection det(float cx, float cy, float s) {
    Detection d;
    d.x0 = cx - s / 2, d.x1 = cx + s / 2, d.y0 = cy - s / 2, d.y1 = cy + s / 2;
    d.score = 0.8f;
    return d;
}

struct Run {
    ClassNames names;
    Tracker trk;
    Nv12Image img;
    std::vector<TrackReport> rep;
    std::vector<EndedTrack> ended;
    uint64_t seq = 0;
    Run(const TrackerConfig &c, const ClassNames &n) : names(n), trk(c, names) {
        img.width = 1280, img.height = 720;
        img.data.assign(size_t(1280) * 720 * 3 / 2, 128);
    }
    void step(const std::vector<Detection> &d, const float *seen = nullptr) {
        ended.clear();
        trk.update(++seq, img.view(), d, rep, ended, seen);
    }
};

ClassNames make_names() {
    const char *p = "/tmp/signs_tracker_test_names.txt";
    std::ofstream(p) << "regulatory--stop\nwarning--other\n";
    ClassNames n;
    n.load(p, nullptr);
    return n;
}

// A confirmed track outside the window of the second view is not aged on that view's frames.
void test_window_does_not_age() {
    TrackerConfig c;
    c.max_age = 2;
    const ClassNames n = make_names();
    Run r(c, n);
    for (int i = 0; i < 3; ++i) r.step({det(100, 100, 20)});
    CHECK(r.rep.size() == 1);
    const float win[4] = {600, 200, 1200, 600};  // the track at (100,100) is outside
    for (int i = 0; i < 4; ++i) r.step({}, win);
    CHECK(r.ended.empty());
    CHECK(r.trk.live() == 1);
    r.step({det(100, 100, 20)});  // seen again by the whole-frame view: same track
    CHECK(r.rep.size() == 1 && r.rep[0].hits == 4);
}

// Without a window the same gap ends the track after max_age (behaviour before L24-S).
void test_no_window_unchanged() {
    TrackerConfig c;
    c.max_age = 2;
    const ClassNames n = make_names();
    Run r(c, n);
    for (int i = 0; i < 3; ++i) r.step({det(100, 100, 20)});
    size_t ended = 0;
    for (int i = 0; i < 4; ++i) r.step({}), ended += r.ended.size();
    CHECK(ended == 1);
    CHECK(r.trk.live() == 0);
}

// Inside the window a missed track ages as before; outside it ends after 3 x max_age frames at the latest.
void test_window_inside_ages_and_cap() {
    TrackerConfig c;
    c.max_age = 2;
    const ClassNames n = make_names();
    Run a(c, n);
    for (int i = 0; i < 3; ++i) a.step({det(800, 300, 20)});
    const float win[4] = {600, 200, 1200, 600};
    size_t ended = 0;
    for (int i = 0; i < 3; ++i) a.step({}, win), ended += a.ended.size();
    CHECK(ended == 1);
    Run b(c, n);
    for (int i = 0; i < 3; ++i) b.step({det(100, 100, 20)});
    ended = 0;
    for (int i = 0; i < 7; ++i) b.step({}, win), ended += b.ended.size();
    CHECK(ended == 1);
}
}  // namespace

int main() {
    test_window_does_not_age();
    test_no_window_unchanged();
    test_window_inside_ages_and_cap();
    if (failures) fprintf(stderr, "signs_tracker_test: %d failure(s)\n", failures);
    else fprintf(stderr, "signs_tracker_test: all passed\n");
    return failures ? 1 : 0;
}
