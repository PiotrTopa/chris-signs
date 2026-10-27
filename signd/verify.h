// Appearance checks for round red sign classes (roadsense lane L8: stop lamps and red traffic lights read
// as `no-entry` and friends). Per detection whose classifier top-1 is a round red class ("RR": prohibitions,
// speed limits, stop; is_round_red()), the box is resampled to 32x32 RGB (same resampler as the classifier)
// and two numbers are computed:
//
//   glow    share of the box's inscribed ellipse that is either blown out (min(R,G,B) >= 235) or "halo"
//           (R >= 220, 80 <= min(R,G,B) < 235, R - min >= 40): a lamp's saturated core fading into bright
//           pink-red. A painted sign has flat dark-ish red (R ~100-200 by day) and a white bar that is not
//           clipped, so its glow is ~0 (Krakow replay: 99 % of real-sign detections < 0.07, every lamp/light
//           detection >= 0.08). But sunlit real signs with a blown-out
//           white field glow too (GTSRB: ~15 % of physical round red signs would be flagged), so by default glow
//           only suppresses when the verifier is unsure as well (TrackerConfig glow_alone, verify_vouch).
//   verify  optional: probability "real sign" from a small CNN (32x32 RGB /255 NCHW in, one logit out),
//           trained by tools/verifier/train_verifier.py on open data (docs/SIGN_MODELS.md "Verifier").
//
// The tracker turns these into a per-track decision (suppressed + reason), see tracker.h.
#pragma once

#include <string>

namespace roadsense {
namespace signs {

constexpr int kAppearanceSize = 32;

// Merged MTSD name (variant suffix stripped) of a round red class.
bool is_round_red(const std::string &merged_name);

// rgb: planar float RGB in [0,1], 3 planes of s*s. Returns the glow share in [0,1].
float glow_score(const float *rgb, int s);

}  // namespace signs
}  // namespace roadsense
