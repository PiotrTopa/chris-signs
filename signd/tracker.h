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
    float cls_gate = 0.6f;     // classifier probability needed for a vote
    float crop_margin = 0.15f; // relative margin around the box for the saved crop
};

struct TrackReport {  // a confirmed track matched in the current frame
    int id;
    std::string cls;
    int cls_id;
    float score;
    float box[4];
    int age, hits;
};

struct EndedTrack {
    int id;
    std::string cls;
    int cls_id;
    uint64_t first_seq, last_seq;
    int hits;
    float best_score;
    Nv12Image crop;
};

class Tracker {
public:
    Tracker(const TrackerConfig &cfg, const ClassNames &names) : cfg_(cfg), names_(names) {}

    // One processed frame. `frame` is used for the best crops.
    void update(uint64_t seq, const Nv12View &frame, const std::vector<Detection> &dets,
                std::vector<TrackReport> &reports, std::vector<EndedTrack> &ended);
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
        bool confirmed = false;
        uint64_t first_seq = 0, last_seq = 0;
        float best_score = 0;
        Nv12Image crop;
        std::vector<float> votes;  // per merged class
        int nvotes = 0;
    };
    void classify(const Track &t, std::string *cls, int *cls_id) const;
    void end(Track &t, std::vector<EndedTrack> &ended);

    TrackerConfig cfg_;
    const ClassNames &names_;
    std::vector<Track> tracks_;
    int next_id_ = 1;
};

}  // namespace signs
}  // namespace roadsense
