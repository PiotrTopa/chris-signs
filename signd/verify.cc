#include "verify.h"

#include <algorithm>

namespace roadsense {
namespace signs {

bool is_round_red(const std::string &n) {
    static const char *const kPrefixes[] = {"regulatory--no-",        "regulatory--maximum-speed-limit",
                                            "regulatory--road-closed", "regulatory--height-limit",
                                            "regulatory--weight-limit", "regulatory--width-limit"};
    for (const char *p : kPrefixes)
        if (n.compare(0, std::char_traits<char>::length(p), p) == 0) return true;
    return n == "regulatory--stop";
}

float glow_score(const float *rgb, int s) {
    const float *R = rgb, *G = rgb + s * s, *B = rgb + 2 * s * s;
    int n = 0, hit = 0;
    const float c = s / 2.0f;
    for (int y = 0; y < s; ++y) {
        const float dy = (y + 0.5f - c) / c;
        for (int x = 0; x < s; ++x) {
            const float dx = (x + 0.5f - c) / c;
            if (dx * dx + dy * dy > 1.0f) continue;
            ++n;
            const int i = y * s + x;
            const float r = R[i] * 255.f, g = G[i] * 255.f, b = B[i] * 255.f;
            const float mn = std::min(r, std::min(g, b));
            if (mn >= 235.f || (r >= 220.f && mn >= 80.f && r - mn >= 40.f)) ++hit;
        }
    }
    return n ? float(hit) / float(n) : 0.f;
}

}  // namespace signs
}  // namespace roadsense
