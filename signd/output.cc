#include "output.h"

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "jpeg_nv12.h"  // camerad's encoder (header-only, no dependencies)

namespace roadsense {
namespace signs {

bool mkdir_p(const std::string &path) {
    std::string cur;
    for (size_t i = 0; i <= path.size(); ++i) {
        if (i == path.size() || path[i] == '/') {
            if (!cur.empty() && mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
        }
        if (i < path.size()) cur.push_back(path[i]);
    }
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool write_file_atomic(const std::string &path, const uint8_t *data, size_t n) {
    const std::string tmp = path + ".tmp";
    int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return false;
    size_t off = 0;
    while (off < n) {
        ssize_t w = write(fd, data + off, n - off);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) {
            close(fd);
            unlink(tmp.c_str());
            return false;
        }
        off += size_t(w);
    }
    close(fd);
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        unlink(tmp.c_str());
        return false;
    }
    return true;
}

bool CropStore::init(std::string *err) {
    if (!mkdir_p(dir_)) {
        if (err) *err = "cannot create " + dir_ + ": " + strerror(errno);
        return false;
    }
    return true;
}

std::string CropStore::save(int track_id, const Nv12Image &img) {
    if (img.width < 2 || img.height < 2) return {};
    static JpegNV12Encoder enc(90);
    const Nv12View v = img.view();
    const std::vector<uint8_t> &jpg = enc.encode(v.y, size_t(v.y_stride), v.uv, size_t(v.uv_stride), img.width,
                                                 img.height);
    const std::string path = dir_ + "/" + std::to_string(track_id) + ".jpg";
    if (!write_file_atomic(path, jpg.data(), jpg.size())) return {};
    written_.push_back(path);
    while (max_ > 0 && written_.size() > max_) {
        unlink(written_.front().c_str());
        written_.pop_front();
    }
    return path;
}

namespace {

void esc(std::string &o, const std::string &s) {
    o.push_back('"');
    for (char c : s) {
        if (c == '"' || c == '\\') {
            o.push_back('\\');
            o.push_back(c);
        } else if (uint8_t(c) < 0x20) {
            char b[8];
            snprintf(b, sizeof(b), "\\u%04x", c);
            o += b;
        } else {
            o.push_back(c);
        }
    }
    o.push_back('"');
}

void num(std::string &o, double v, int prec) {
    char b[32];
    snprintf(b, sizeof(b), "%.*f", prec, v);
    o += b;
}

}  // namespace

std::string signs_json(const FrameMeta &m, double detect_ms, const std::vector<TrackReport> &dets,
                       const std::vector<EndedTrack> &ended, const std::vector<std::string> &crop_paths,
                       const ExtraFields &extra) {
    std::string o;
    o.reserve(256 + 160 * (dets.size() + ended.size()));
    char b[160];
    snprintf(b, sizeof(b), "{\"type\":\"signs\",\"seq\":%" PRIu64 ",\"ts_mono_ns\":%" PRId64 ",\"ts_real_ns\":%" PRId64
             ",\"detect_ms\":",
             m.seq, m.ts_mono_ns, m.ts_real_ns);
    o += b;
    num(o, detect_ms, 1);
    o += ",\"detections\":[";
    for (size_t i = 0; i < dets.size(); ++i) {
        const TrackReport &d = dets[i];
        if (i) o.push_back(',');
        o += "{\"track_id\":" + std::to_string(d.id) + ",\"cls\":";
        esc(o, d.cls);
        o += ",\"cls_id\":" + std::to_string(d.cls_id) + ",\"score\":";
        num(o, d.score, 3);
        o += ",\"box\":[";
        for (int k = 0; k < 4; ++k) {
            if (k) o.push_back(',');
            num(o, d.box[k], 1);
        }
        o += "],\"age\":" + std::to_string(d.age) + ",\"hits\":" + std::to_string(d.hits);
        if (d.glow_share >= 0) o += ",\"glow\":", num(o, d.glow_share, 2);
        if (d.verify >= 0) o += ",\"verify\":", num(o, d.verify, 2);
        if (d.suppressed) o += ",\"suppressed\":true,\"reason\":\"" + std::string(d.reason) + "\"";
        o.push_back('}');
    }
    o += "],\"tracks_ended\":[";
    for (size_t i = 0; i < ended.size(); ++i) {
        const EndedTrack &e = ended[i];
        if (i) o.push_back(',');
        o += "{\"track_id\":" + std::to_string(e.id) + ",\"cls\":";
        esc(o, e.cls);
        snprintf(b, sizeof(b), ",\"first_seq\":%" PRIu64 ",\"last_seq\":%" PRIu64 ",\"hits\":%d,\"best_score\":",
                 e.first_seq, e.last_seq, e.hits);
        o += b;
        num(o, e.best_score, 3);
        o += ",\"best_crop\":";
        if (i < crop_paths.size() && !crop_paths[i].empty())
            esc(o, crop_paths[i]);
        else
            o += "null";
        if (e.glow_share >= 0) o += ",\"glow\":", num(o, e.glow_share, 2);
        if (e.verify >= 0) o += ",\"verify\":", num(o, e.verify, 2);
        if (e.suppressed) o += ",\"suppressed\":true,\"reason\":\"" + std::string(e.reason) + "\"";
        o.push_back('}');
    }
    o.push_back(']');
    if (extra.timing) {
        const Timing &t = *extra.timing;
        o += ",\"timing\":{\"pre_ms\":";
        num(o, t.pre_ms, 1);
        o += ",\"det_ms\":";
        num(o, t.det_ms, 1);
        o += ",\"post_ms\":";
        num(o, t.post_ms, 2);
        o += ",\"cls_ms\":";
        num(o, t.cls_ms, 1);
        o.push_back('}');
    }
    if (extra.view >= 0) o += ",\"view\":" + std::to_string(extra.view);
    if (extra.raw && extra.names) {
        o += ",\"raw\":[";
        for (size_t i = 0; i < extra.raw->size(); ++i) {
            const Detection &d = (*extra.raw)[i];
            if (i) o.push_back(',');
            o += "{\"cls\":";
            esc(o, d.cls_raw >= 0 ? extra.names->raw[size_t(d.cls_raw)] : std::string("traffic-sign"));
            o += ",\"score\":";
            num(o, d.score, 3);
            o += ",\"cls_prob\":";
            num(o, d.cls_prob, 3);
            if (d.glow >= 0) o += ",\"glow\":", num(o, d.glow, 3);
            if (d.verify >= 0) o += ",\"verify\":", num(o, d.verify, 3);
            o += ",\"box\":[";
            const float bx[4] = {d.x0, d.y0, d.x1, d.y1};
            for (int k = 0; k < 4; ++k) {
                if (k) o.push_back(',');
                num(o, bx[k], 1);
            }
            o += "]}";
        }
        o.push_back(']');
    }
    o.push_back('}');
    return o;
}

}  // namespace signs
}  // namespace roadsense
