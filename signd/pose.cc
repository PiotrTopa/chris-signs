#include "pose.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "lights.h"

namespace roadsense {
namespace signs {

const char *pose_view_name(int c) {
    static const char *n[4] = {"rear", "front", "side", "threequarter"};
    return c >= 0 && c < 4 ? n[c] : "unknown";
}

bool ru_pose_input(const Nv12View &f, const RoadUserDet &u, int cut, const PoseCam &cam, float *in) {
    const double bw = double(u.x1) - u.x0, bh = double(u.y1) - u.y0;
    if (bw < 8.0 || bh < 8.0) return false;
    constexpr int S = kPoseS;
    constexpr size_t plane = size_t(S) * S;
    std::fill(in, in + 3 * plane, 0.f);
    // the crop: the box scaled about its centre; crop pixel j covers [rx0 + j px, rx0 + (j + 1) px) of the frame
    const double cx = 0.5 * (double(u.x0) + u.x1), cy = 0.5 * (double(u.y0) + u.y1);
    const double rw = bw * kPoseMargin, rh = bh * kPoseMargin, rx0 = cx - 0.5 * rw, ry0 = cy - 0.5 * rh;
    const double px = rw / S, py = rh / S;
    // the crop pixels that lie in the picture (the others stay black)
    const int j0 = std::max(0, int(std::ceil((0.0 - rx0) / px - 1e-9))), j1 = std::min(S, int(std::floor((f.width - rx0) / px + 1e-9)));
    const int i0 = std::max(0, int(std::ceil((0.0 - ry0) / py - 1e-9))), i1 = std::min(S, int(std::floor((f.height - ry0) / py + 1e-9)));
    if (j1 > j0 && i1 > i0)
        nv12_region_to_rgb_planar(f, rx0 + j0 * px, ry0 + i0 * py, (j1 - j0) * px, (i1 - i0) * py, j1 - j0, i1 - i0, in, S, plane, j0, i0);
    if (u.x0 <= 2.f) cut |= 1;
    if (u.y0 <= 2.f) cut |= 2;
    if (u.x1 >= float(f.width) - 2.f) cut |= 4;
    if (u.y1 >= float(f.height) - 2.f) cut |= 8;
    float g[kPoseG] = {float(std::log(bw / bh)), (cut & 1) ? 1.f : 0.f, (cut & 2) ? 1.f : 0.f, (cut & 4) ? 1.f : 0.f,
                       (cut & 8) ? 1.f : 0.f, 0.f, 0.f, 0.f, 0.f};
    if (cam.ok && cam.f > 1.f) {
        g[5] = 1.f;
        g[6] = float((cx - cam.cu) / cam.f);
        g[7] = float((double(u.y1) - cam.cv) / cam.f);
        g[8] = float(std::log(bh / cam.f) + 3.0);
    }
    for (int k = 0; k < kPoseG; ++k) std::fill(in + (3 + k) * plane, in + (4 + k) * plane, g[k]);
    return true;
}

bool ru_pose_decode(const float *o, const RoadUserDet &u, RuPose &p) {
    p = RuPose();
    for (int i = 0; i < kPoseOut; ++i)
        if (!std::isfinite(o[i])) return false;
    const float bw = u.x1 - u.x0, bh = u.y1 - u.y0, cx = 0.5f * (u.x0 + u.x1);
    const auto pr = [](float v) { return std::min(1.f, std::max(0.f, v)); };
    p.yaw = o[0];
    p.yaw_std = std::max(0.f, o[1]);
    p.p_main = pr(o[2]), p.p_flip = pr(o[3]);
    for (int i = 0; i < 4; ++i) p.vp[i] = pr(o[4 + i]);
    p.gc[0] = cx + o[8] * bw, p.gc[1] = u.y1 + o[9] * bh;
    p.gn_y = u.y1 + o[10] * bh;
    p.gl[0] = cx + o[11] * bw, p.gl[1] = u.y1 + o[12] * bh;
    p.gr[0] = cx + o[13] * bw, p.gr[1] = u.y1 + o[14] * bh;
    p.gc_std[0] = std::fabs(o[15]) * bw, p.gc_std[1] = std::fabs(o[16]) * bh;
    p.gn_std = std::fabs(o[17]) * bh;
    p.gl_std[0] = std::fabs(o[18]) * bw, p.gl_std[1] = std::fabs(o[19]) * bh;
    p.gr_std[0] = std::fabs(o[20]) * bw, p.gr_std[1] = std::fabs(o[21]) * bh;
    p.gc_vis = pr(o[22]), p.gn_vis = pr(o[23]), p.gl_vis = pr(o[24]), p.gr_vis = pr(o[25]);
    for (int i = 0; i < 3; ++i) p.dim[i] = std::max(0.f, o[26 + i]), p.dim_std[i] = std::fabs(o[29 + i]);
    p.occ = pr(o[32]), p.occ2 = pr(o[33]), p.trunc = pr(o[34]);
    for (int i = 0; i < 3; ++i) p.kind[i] = pr(o[35 + i]);
    p.ok = true;
    return true;
}

std::string pose_json(const RuPose &p) {
    if (!p.ok) return std::string();
    const int c = int(std::max_element(p.vp, p.vp + 4) - p.vp);
    char b[640];
    const auto F = [](double v, int d) {
        char t[32];
        snprintf(t, sizeof(t), "%.*f", d, v);
        return std::string(t);
    };
    snprintf(b, sizeof(b),
             ",\"pose\":{\"yaw\":%s,\"yaw_std\":%s,\"p\":%s,\"flip\":%s,\"vp\":\"%s\",\"vp_conf\":%s,"
             "\"gc\":[%s,%s,%s,%s,%s],\"gn\":[%s,%s,%s,%s],\"gl\":[%s,%s,%s,%s,%s],\"gr\":[%s,%s,%s,%s,%s],"
             "\"dim\":[%s,%s,%s],\"dim_std\":[%s,%s,%s],\"occ\":%s,\"trunc\":%s,\"kind\":[%s,%s,%s]}",
             F(p.yaw, 1).c_str(), F(p.yaw_std, 1).c_str(), F(p.p_main, 2).c_str(), F(p.p_flip, 2).c_str(), pose_view_name(c),
             F(p.vp[c], 2).c_str(),
             F(p.gc[0], 1).c_str(), F(p.gc[1], 1).c_str(), F(p.gc_std[0], 1).c_str(), F(p.gc_std[1], 1).c_str(), F(p.gc_vis, 2).c_str(),
             F(p.gc[0], 1).c_str(), F(p.gn_y, 1).c_str(), F(p.gn_std, 1).c_str(), F(p.gn_vis, 2).c_str(),
             F(p.gl[0], 1).c_str(), F(p.gl[1], 1).c_str(), F(p.gl_std[0], 1).c_str(), F(p.gl_std[1], 1).c_str(), F(p.gl_vis, 2).c_str(),
             F(p.gr[0], 1).c_str(), F(p.gr[1], 1).c_str(), F(p.gr_std[0], 1).c_str(), F(p.gr_std[1], 1).c_str(), F(p.gr_vis, 2).c_str(),
             F(p.dim[0], 2).c_str(), F(p.dim[1], 2).c_str(), F(p.dim[2], 2).c_str(), F(p.dim_std[0], 2).c_str(), F(p.dim_std[1], 2).c_str(),
             F(p.dim_std[2], 2).c_str(), F(p.occ, 2).c_str(), F(p.trunc, 2).c_str(), F(p.kind[0], 2).c_str(), F(p.kind[1], 2).c_str(),
             F(p.kind[2], 2).c_str());
    return b;
}

}  // namespace signs
}  // namespace roadsense
