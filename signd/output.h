// signd output: the `signs` JSON line (roadsense docs/ARCHITECTURE.md) and best-crop
// JPEG files. Shared by signd and tools/signd_replay.
#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "tracker.h"
#include "yolo.h"

namespace roadsense {
namespace signs {

struct FrameMeta {
    uint64_t seq = 0;
    int64_t ts_mono_ns = 0, ts_real_ns = 0;
};

// Writes <dir>/<track_id>.jpg (temp file + rename) and keeps at most
// max_files crops written by this process (oldest deleted first).
class CropStore {
public:
    CropStore(std::string dir, size_t max_files) : dir_(std::move(dir)), max_(max_files) {}
    bool init(std::string *err);                                   // mkdir -p
    std::string save(int track_id, const Nv12Image &img);          // returns the path, "" on failure
    const std::string &dir() const { return dir_; }

private:
    std::string dir_;
    size_t max_;
    std::deque<std::string> written_;
};

// Optional extra per-frame fields (not in the contract; off in signd unless
// --timing is given).
struct ExtraFields {
    const Timing *timing = nullptr;
    const std::vector<Detection> *raw = nullptr;  // all detections of this frame, before tracking
    const ClassNames *names = nullptr;
    int view = -1;  // >= 0: which detector view this frame used (only with a second view, --roi2)
};

std::string signs_json(const FrameMeta &m, double detect_ms, const std::vector<TrackReport> &dets,
                       const std::vector<EndedTrack> &ended, const std::vector<std::string> &crop_paths,
                       const ExtraFields &extra = {});

bool mkdir_p(const std::string &path);
bool write_file_atomic(const std::string &path, const uint8_t *data, size_t n);

}  // namespace signs
}  // namespace roadsense
