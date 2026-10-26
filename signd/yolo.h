// signd pre/post-processing and inference: MTSD two-stage traffic-sign
// pipeline (docs/SIGN_MODELS.md).
//
//   NV12 frame (any even WxH, full-range BT.601, see camerad/ring.h)
//     -> Y and CbCr planes resampled straight to the letterbox size
//        (separable triangle filter = PIL BILINEAR with antialiasing, the
//        reference implementation), YCbCr -> RGB float NCHW /255 on the small
//        image only, grey 114 padding
//     -> YOLOv8n single-class detector [1,3,384,640] -> [1,5,5040]
//     -> score > det_thr, greedy NMS (IoU nms_iou), max 100, map back to frame px
//     -> per box: Ultralytics classify transform (short side to 64, centre
//        crop) sampled directly from the full-resolution NV12 frame
//     -> YOLOv8n-cls [1,3,64,64] -> 400 softmax probabilities
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "runner.h"

namespace roadsense {
namespace signs {

// Read-only view of an NV12 image (Y plane + interleaved CbCr plane, Cb first).
struct Nv12View {
    const uint8_t *y = nullptr;
    const uint8_t *uv = nullptr;
    int width = 0, height = 0;  // even
    int y_stride = 0, uv_stride = 0;
};

// Owned NV12 image (used for the best crop of a track and by tools).
struct Nv12Image {
    int width = 0, height = 0;  // even
    std::vector<uint8_t> data;  // Y (width*height) then CbCr (width*height/2), strides = width
    Nv12View view() const {
        Nv12View v;
        v.y = data.data();
        v.uv = data.data() + size_t(width) * height;
        v.width = width;
        v.height = height;
        v.y_stride = v.uv_stride = width;
        return v;
    }
};

struct PipelineConfig {
    // Backends per model (runner.h): "ort" (ONNX Runtime CPU, the reference) or "ncnn" (Vulkan).
    // Empty model paths resolve to the default file names for the backend in models_dir
    // (ort: mtsd_signdet_yolov8n_384x640.onnx / mtsd_signcls_yolov8n_64.onnx, ncnn: the same
    // stems with -ncnn.param), see default_model_path().
    std::string det_backend = "ort", cls_backend = "ort";
    std::string models_dir = "/var/lib/roadsense/models";
    std::string det_model, cls_model;
    std::string cls_names;     // default models_dir/mtsd_signcls_classes.txt
    int threads = 4;           // ORT intra-op threads (both sessions); ncnn CPU threads
    bool ncnn_fp16 = true;     // ncnn: fp16-packed storage (false: GPU fp32, several times slower)
    bool ncnn_flushall = true; // ncnn: add TU_DEBUG=flushall for turnip (see runner.h)
    // Model geometry: ORT checks these against the graph, ncnn (no shapes in .param) relies on them.
    int det_w = 640, det_h = 384, cls_size = 64;
    float det_thr = 0.35f;     // detector score threshold (strictly greater)
    float cls_gate = 0.6f;     // classifier top-1 probability needed for a class vote
    float nms_iou = 0.45f;
    int max_det = 100;
    float cls_margin = 0.0f;   // relative box padding before the classifier crop (reference: 0)
    float min_box_px = 4.0f;   // drop degenerate boxes (frame px) before classifying
    // ORT intra-op busy-wait between ops (session.intra_op.allow_spinning). Off by
    // default: on the SD845 spinning made the detector slower (A55: 422 vs 323 ms,
    // A75: 183 vs 145 ms, 4 threads) and burns cores camerad shares.
    bool allow_spinning = false;
};

// models_dir/<default file name> for a model ("det" or "cls") on a backend.
std::string default_model_path(const std::string &models_dir, const std::string &which, const std::string &backend);

struct Detection {
    float x0, y0, x1, y1;   // frame pixels
    float score;            // detector score
    int cls_raw = -1;       // classifier argmax (line in the classes file), -1 if not classified
    float cls_prob = 0.f;   // its probability
    std::vector<float> probs;  // full classifier output (empty if not classified)
};

struct Timing {
    double pre_ms = 0, det_ms = 0, post_ms = 0, cls_ms = 0, total_ms = 0;
    double wait_ms = 0;  // time spent in the before-detector hook (GPU schedule), included in total_ms
};

// Class names: raw MTSD names (with --gN variant suffix) and the merged label
// set used for reporting (suffix stripped, e.g. regulatory--maximum-speed-limit-50).
struct ClassNames {
    std::vector<std::string> raw;     // 400
    std::vector<std::string> merged;  // unique stripped names, in order of first appearance
    std::vector<int> raw_to_merged;   // raw id -> merged index
    std::vector<int> merged_to_raw;   // merged index -> first raw id (reported as cls_id)
    bool load(const std::string &path, std::string *err);
};

class SignPipeline {
public:
    explicit SignPipeline(const PipelineConfig &cfg);  // throws std::runtime_error
    ~SignPipeline();
    SignPipeline(const SignPipeline &) = delete;
    SignPipeline &operator=(const SignPipeline &) = delete;

    // Full pipeline on one frame. Detections are sorted by score.
    void process(const Nv12View &frame, std::vector<Detection> &out, Timing *t = nullptr);

    const ClassNames &names() const { return names_; }
    const PipelineConfig &config() const { return cfg_; }  // with the model paths resolved
    int input_w() const { return in_w_; }
    int input_h() const { return in_h_; }
    const char *det_backend() const { return det_->name(); }
    const char *cls_backend() const { return cls_->name(); }
    // Optional hooks around the detector run (signd: GPU turn-taking with modeld, roadsense
    // docs/ARCHITECTURE.md "GPU schedule"). before_det runs after the letterbox, at the last moment before
    // the detector's GPU work is submitted (ncnn: once its command buffer is recorded, ModelRunner::run_hooked);
    // after_det right after the detector returns. Timing::wait_ms is the time spent in before_det.
    std::function<void()> before_det, after_det;

private:
    std::unique_ptr<ModelRunner> det_, cls_;
    PipelineConfig cfg_;
    ClassNames names_;
    int lb_w_ = -1, lb_h_ = -1;  // letterbox content size of the last frame
    int in_w_ = 640, in_h_ = 384, n_anchor_ = 0, n_out_ = 0, cls_s_ = 64, n_cls_ = 0;
    std::vector<float> det_in_, det_out_, cls_in_, cls_out_;
};

// ---- image helpers (also used by tools/signd_replay) ----------------------

// Resamples the source rectangle [sx, sx+sw) x [sy, sy+sh) (sub-pixel, in
// plane pixels) of the Y plane and the matching chroma rectangle to dw x dh,
// converts YCbCr -> RGB and writes float planar RGB /255 into dst (3 planes of
// dst_plane floats, row pitch dst_w floats, starting at (dx, dy)).
void nv12_region_to_rgb_planar(const Nv12View &src, double sx, double sy, double sw, double sh, int dw, int dh,
                               float *dst, int dst_w, size_t dst_plane, int dx, int dy);

// Copies an even-aligned NV12 sub-rectangle (clipped to the frame).
void nv12_crop(const Nv12View &src, int x0, int y0, int x1, int y1, Nv12Image &out);

// Full-range BT.601 conversions (tools).
void rgb_to_nv12(const uint8_t *rgb, int width, int height, Nv12Image &out);  // crops to even size
void nv12_to_rgb(const Nv12View &src, std::vector<uint8_t> &rgb);            // nearest chroma

std::string strip_variant(const std::string &name);  // "...--g1" -> "..."

}  // namespace signs
}  // namespace roadsense
