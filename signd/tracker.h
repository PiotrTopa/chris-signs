// SORT-style multi-object tracker for traffic signs (roadsense docs/ARCHITECTURE.md).
//
// Per track a constant-velocity Kalman filter on (cx, cy, area, aspect) as in
// SORT (Bewley et al. 2016): state [cx, cy, s, r, vcx, vcy, vs], one step per
// PROCESSED frame. signd always takes the newest ring frame, so processed
// frames are close to evenly spaced at its own rate (2-10 fps); max_age is
// counted in processed frames too, so a unit step keeps both consistent.
// Process noise on the velocities is raised over SORT's 30 fps defaults
// because at a few fps an approaching sign moves and grows a lot per step.
//
// Association: greedy on IoU(prediction, detection) >= iou_match (sparse
// scenes, a handful of boxes: greedy and Hungarian agree except in contrived
// ties), then a second greedy pass for the leftovers on centre distance
// normalised by the box size (catches small fast signs whose boxes no longer
// overlap at a low frame rate; the velocity estimate needs two hits first).
//
// Per track it accumulates class votes (sum over frames of the classifier
// probabilities, only from frames whose top-1 probability passed the gate,
// merged over --gN graphical variants), hits, first/last seq, best detector
// score and the NV12 crop taken at that best score.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "yolo.h"

namespace roadsense {
namespace signs {

struct TrackerConfig {
    int max_age = 10;          // processed frames without a match before a track dies
    int min_hits = 3;          // matched detections before a track is confirmed (reported)
    float iou_match = 0.1f;    // min IoU for the first association pass
    float dist_match = 1.0f;   // max centre distance / max(w,h) for the fallback pass
    float dist_match_new = 1.0f;  // the same for a track with one hit (no velocity yet): at 5 Hz a small sign at
                                  // the frame edge moves more than its own width between processed frames (L12)
    float cls_gate = 0.6f;     // classifier probability needed for a vote
    float crop_margin = 0.15f; // relative margin around the box for the saved crop
    // Acceptance (roadsense lane L8). A confirmed track is still reported, but flagged `suppressed` with a
    // `reason` when its current class is a round red one (verify.h) and
    //   verifier: >= verify_min verified detections with a mean probability < verify_thr, or
    //   glow:     >= glow_min checked detections and a share >= glow_share of them had glow >= glow_px, and the
    //             verifier is not sure either (mean < verify_vouch); without a verifier glow only suppresses
    //             with glow_alone (off by default: by itself it also flags ~15 % of sunlit real signs, GTSRB);
    // or (any class) its best detector score is below the class's minimum (class_min_score, prefix match on
    // the merged name, longest prefix wins): reason "score" (can clear later as the sign gets closer).
    float glow_px = 0.10f;
    float glow_share = 0.5f;
    int glow_min = 2;
    bool glow_alone = false;
    float verify_vouch = 0.9f;
    float verify_thr = 0.3f;
    int verify_min = 2;
    std::vector<std::pair<std::string, float>> class_min_score;
    // External boxes (Detection::external, signd --lights-signs) may start tracks of their own (true) or only
    // continue tracks the sign detector started (false, --lights-signs-keep).
    bool external_starts = true;
    // External boxes count as hits (confirmation, the reported `hits`) (true), or only keep a track alive and
    // placed and add their class votes (false): `hits` then keeps its meaning of sign-detector matches.
    bool external_hits = true;
};

// "prefix=0.7,prefix2=0.5" -> class_min_score; false on a syntax error.
bool parse_class_min_score(const std::string &s, std::vector<std::pair<std::string, float>> *out);

struct TrackReport {  // a confirmed track matched in the current frame
    int id;
    std::string cls;
    int cls_id;
    float score;
    float box[4];
    int age, hits;
    bool suppressed = false;
    const char *reason = nullptr;  // "glow" | "verifier" | "score" when suppressed
    float glow_share = -1.f;       // share of checked detections that glowed (-1: none checked)
    float verify = -1.f;           // mean verifier probability over the track (-1: none verified)
};

struct EndedTrack {
    int id;
    std::string cls;
    int cls_id;
    uint64_t first_seq, last_seq;
    int hits;
    float best_score;
    Nv12Image crop;
    bool suppressed = false;
    const char *reason = nullptr;
    float glow_share = -1.f;
    float verify = -1.f;
};

class Tracker {
public:
    Tracker(const TrackerConfig &cfg, const ClassNames &names);

    // One processed frame. `frame` is used for the best crops. `seen` (optional, frame pixels x0, y0, x1, y1):
    // the part of the frame the detector looked at this time (second view, roadsense lane L24-S); a track whose
    // predicted centre lies outside it could not have been matched and is not counted as missed (it still
    // ends after 3 x max_age frames without a match).
    void update(uint64_t seq, const Nv12View &frame, const std::vector<Detection> &dets,
                std::vector<TrackReport> &reports, std::vector<EndedTrack> &ended, const float *seen = nullptr);
    // Ends all live tracks (shutdown / end of a replay sequence).
    void flush(std::vector<EndedTrack> &ended);

    size_t live() const { return tracks_.size(); }

private:
    struct Kalman {
        double x[7];
        double P[7][7];
        void init(const float box[4]);
        void predict();
        void update(const float box[4]);
        void box(float out[4]) const;
    };
    struct Track {
        int id;
        Kalman kf;
        float pred[4];
        int age = 0, hits = 0, since_update = 0;
        int unseen = 0;  // frames without a match in which the track was outside the view (since_update excludes them)
        bool confirmed = false;
        uint64_t first_seq = 0, last_seq = 0;
        float best_score = 0;
        Nv12Image crop;
        std::vector<float> votes;  // per merged class
        int nvotes = 0;
        int n_glow_checked = 0, n_glow = 0, n_verified = 0;
        float verify_sum = 0;
    };
    void classify(const Track &t, std::string *cls, int *cls_id) const;
    // Acceptance decision for the track's current class (merged index k, -1 = unclassified).
    const char *suppress_reason(const Track &t, int k) const;
    int merged_index(const Track &t) const;
    void end(Track &t, std::vector<EndedTrack> &ended);

    TrackerConfig cfg_;
    const ClassNames &names_;
    std::vector<Track> tracks_;
    std::vector<char> red_merged_;
    std::vector<float> min_score_merged_;
    int next_id_ = 1;
};

}  // namespace signs
}  // namespace roadsense
