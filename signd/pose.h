// Pose of a road user from a trained head (signd --lights-ru-pose FILE.onnx; docs/POSE_HEAD.md): which way the
// object points and where it stands on the road. One run of a small network per box crop, the same run-time path
// as the learnt appearance vector (ONNX Runtime, one thread). signd holds only the machinery: how the crop and
// the box numbers are fed, how the model's outputs are read and published. The model file is trained and
// distributed elsewhere (chris-pose) under its own licence, is loaded at run time and is not needed to build
// or to pass the tests; without the file the option is off (a line on stderr).
//
// The model contract "pose1":
//   input  [1, 12, 64, 64]: planes 0-2 = the box scaled x 1.3 about its centre, stretched to 64 x 64, RGB / 255,
//          black outside the picture; planes 3-11 = nine numbers about the box, each as a constant plane:
//            ln(box width / box height); 1 where the left / top / right / bottom edge is cut (clip or trunc, or
//            within 2 px of the frame's edge); 1 when the camera numbers are given (--lights-ru-pose-cam), else 0 and
//            the next three 0; (box centre x - cu) / f; (box bottom - cv) / f; ln(box height / f) + 3
//   output [1, 38] (kPoseOut), positions in box units (x from the box centre in box widths, y from the box's
//          bottom edge in box heights, down positive):
//            0 yaw (deg), 1 its 1-sigma (deg), 2 p_main, 3 p_flip, 4-7 view class probabilities (rear, front, side,
//            three-quarter), 8-14 gc x, gc y, gn y, gl x, gl y, gr x, gr y, 15-21 their 1-sigma, 22-25 probability
//            that gc, gn, gl, gr are seen, 26-28 length, width, height (m), 29-31 1-sigma of their logarithm,
//            32 P(occluded), 33 P(largely occluded), 34 share outside the picture, 35-37 P(vehicle, person,
//            two-wheeler)
// yaw: the object's heading against the line of sight from the camera, clockwise seen from above: 0 = it points
// away (its rear is seen), +-180 = it points at the camera, +90 = it points to the right (its right side is seen).
// Ground points (where the object's footprint, the rectangle length x width under it, meets the road): gc its
// centre; gn where the line of sight to the centre enters the rectangle (for a vehicle straight ahead the middle of
// its rear face on the road; the same image column as gc); gl / gr the leftmost / rightmost in the picture of the
// corners on the sides that face the camera.
#pragma once

#include <string>

#include "yolo.h"

namespace roadsense {
namespace signs {

struct RoadUserDet;

constexpr int kPoseS = 64, kPoseG = 9, kPoseOut = 38;
constexpr float kPoseMargin = 1.3f;

struct PoseCam {
    bool ok = false;
    float f = 0.f, cu = 0.f, cv = 0.f;  // focal length and principal point, frame px
};

struct RuPose {
    bool ok = false;
    float yaw = 0, yaw_std = 0, p_main = 0, p_flip = 0;
    float vp[4] = {0, 0, 0, 0};          // rear, front, side, three-quarter
    // ground points, frame px: position, 1-sigma, probability that the point is seen
    float gc[2] = {0, 0}, gc_std[2] = {0, 0}, gc_vis = 0;
    float gn_y = 0, gn_std = 0, gn_vis = 0;   // gn's column is gc's
    float gl[2] = {0, 0}, gl_std[2] = {0, 0}, gl_vis = 0;
    float gr[2] = {0, 0}, gr_std[2] = {0, 0}, gr_vis = 0;
    float dim[3] = {0, 0, 0}, dim_std[3] = {0, 0, 0};  // length, width, height (m); 1-sigma of the logarithm
    float occ = 0, occ2 = 0, trunc = 0;
    float kind[3] = {0, 0, 0};           // vehicle, person, two-wheeler
};

const char *pose_view_name(int c);  // "rear", "front", "side", "threequarter"

// The model's input for a box (12 x 64 x 64 floats); false for a box under 8 px on a side. `cut`: the edges of the
// box that are cut, bits as RoadUserDet::clip (the caller passes clip | trunc; edges within 2 px of the frame's
// edge are added here).
bool ru_pose_input(const Nv12View &f, const RoadUserDet &u, int cut, const PoseCam &cam, float *in);
// The model's 38 outputs for that box -> frame pixels and probabilities. false (out.ok false) when a number is
// not finite.
bool ru_pose_decode(const float *out, const RoadUserDet &u, RuPose &pose);
// ,"pose":{...} for a road user's JSON object.
std::string pose_json(const RuPose &p);

}  // namespace signs
}  // namespace roadsense
