// road_users_test: checks of the road-user part of signd/lights.h that need no model: rider pairing and the
// `road_users` key of the `lights` message (left out when off, null when shed). Exit code 0 = all passed.
//   cmake --build build --target road_users_test && build/road_users_test
#include <cstdio>
#include <string>
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

RoadUserDet box(int cls, float x0, float y0, float x1, float y1, float score) {
    RoadUserDet d;
    d.x0 = x0, d.y0 = y0, d.x1 = x1, d.y1 = y1, d.score = score, d.cls = cls;
    for (float &p : d.p) p = 0.f;
    d.p[cls] = score;
    return d;
}
}  // namespace

int main() {
    {   // a cyclist: rider above the bike, overlapping -> one bike with a rider, union box
        std::vector<RoadUserDet> v = {box(1, 100, 50, 130, 120, 0.7f), box(7, 95, 90, 140, 140, 0.5f), box(2, 300, 80, 400, 150, 0.9f)};
        pair_riders(v);
        CHECK(v.size() == 2);
        CHECK(v[0].cls == 7 && v[0].rider && v[0].y0 == 50 && v[0].y1 == 140 && v[0].x1 == 140 && v[0].score > 0.69f);
        CHECK(v[1].cls == 2 && !v[1].rider);
    }
    {   // a rider beside a parked bike (no horizontal overlap) stays a rider; the bike stays riderless
        std::vector<RoadUserDet> v = {box(1, 100, 50, 130, 120, 0.7f), box(7, 200, 90, 240, 140, 0.5f)};
        pair_riders(v);
        CHECK(v.size() == 2 && v[0].cls == 1 && !v[1].rider);
    }
    {   // two riders, one motorbike: the better overlap gets it
        std::vector<RoadUserDet> v = {box(1, 100, 40, 140, 110, 0.6f), box(1, 112, 40, 152, 110, 0.6f), box(6, 110, 80, 150, 140, 0.8f)};
        pair_riders(v);
        CHECK(v.size() == 2);
        int bikes = 0, riders = 0;
        for (auto &d : v) bikes += d.cls == 6 && d.rider, riders += d.cls == 1;
        CHECK(bikes == 1 && riders == 1);
    }
    {   // message key: off -> absent, shed -> null, on -> list with clip letters
        LightsConfig cfg;
        std::vector<LightTrackReport> none;
        std::vector<RoadUserDet> u = {box(0, 10, 20, 30, 80, 0.8f)};
        u[0].clip = 8 | 1;
        const std::string off = lights_json(1, 2, 3, 1.0, 1280, 720, cfg, none);
        const std::string shed = lights_json(1, 2, 3, 1.0, 1280, 720, cfg, none, nullptr, nullptr, 0, &u, 2);
        const std::string on = lights_json(1, 2, 3, 1.0, 1280, 720, cfg, none, nullptr, nullptr, 0, &u, 1);
        CHECK(off.find("road_users") == std::string::npos);
        CHECK(shed.find("\"road_users\":null") != std::string::npos);
        CHECK(on.find("\"road_users\":[{\"cls\":\"person\",\"score\":0.800,\"box\":[10.0,20.0,30.0,80.0],\"clip\":\"lb\"") != std::string::npos);
    }
    {   // the road-user view's own message
        std::vector<RoadUserDet> u = {box(2, 100, 400, 200, 480, 0.9f)};
        const float roi[4] = {0.f, 0.33f, 1.f, 0.95f};
        const std::string m = road_users_json(7, 8, 9, 50.0, 1280, 720, roi, u);
        CHECK(m.rfind("{\"type\":\"road_users\",\"seq\":7,\"ts_mono_ns\":8,\"ts_real_ns\":9,\"ms\":50.0,\"frame\":[1280,720],"
                      "\"roi\":[0,238,1280,684],\"view\":2,\"road_users\":[{\"cls\":\"car\"", 0) == 0);
        CHECK(m.back() == '}' && m[m.size() - 2] == ']');
    }
    {   // the idle announcement of the light thread (--lights-idle)
        CHECK(lights_idle_hz("ZW.Z.W", 5.0) > 3.33 && lights_idle_hz("ZW.Z.W", 5.0) < 3.34);
        CHECK(lights_idle_hz("W...Z...", 5.0) == 1.25);
        CHECK(lights_idle_hz("", 5.0) == 0);
        LightsConfig cfg;
        std::vector<LightTrackReport> none;
        std::string m = lights_json(1, 2, 3, 1.0, 1280, 720, cfg, none);
        CHECK(m.find("idle") == std::string::npos);
        add_lights_idle(m, "ZW.Z.W", lights_idle_hz("ZW.Z.W", 5.0));
        const std::string tail = ",\"idle\":{\"pattern\":\"ZW.Z.W\",\"hz\":3.33}}";
        CHECK(m.size() > tail.size() && m.compare(m.size() - tail.size(), tail.size(), tail) == 0);
    }
    if (failures) fprintf(stderr, "road_users_test: %d failure(s)\n", failures);
    else fprintf(stderr, "road_users_test: all passed\n");
    return failures ? 1 : 0;
}
