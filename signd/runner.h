// Inference runners for the two sign models (detector and classifier). A runner owns one network
// and runs it on a single fp32 NCHW input (batch 1); the pipeline around it (letterbox, decode,
// NMS, crops, tracker) is shared by all backends and lives in yolo.cc.
//
//   ort   ONNX Runtime, CPU EP, the exported .onnx graphs. Always built; the reference.
//   ncnn  ncnn with Vulkan (fp16-packed storage, fp32 arithmetic on turnip), the converted graphs
//         <stem>.param/.bin from tools/ncnn/make_ncnn.sh. Built with -DROADSENSE_NCNN=ON.
//         Needs TU_DEBUG=flushall on Mesa turnip (set by the runner unless disabled), see
//         roadsense docs/HARDWARE.md.
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace roadsense {
namespace signs {

struct RunnerOptions {
    std::string backend = "ort";  // "ort" | "ncnn"
    std::string model;            // ort: .onnx; ncnn: .param (weights in the .bin next to it)
    int threads = 4;              // ort: intra-op threads; ncnn: CPU threads (none of our layers need them)
    bool allow_spinning = false;  // ort only
    bool ncnn_fp16 = true;        // ncnn: fp16-packed storage (false: fp32 on the GPU, several times slower)
    bool ncnn_flushall = true;    // ncnn: add "flushall" to TU_DEBUG before Vulkan starts (turnip needs it)
    // Expected input [1,C,H,W] and output element count; ort checks them against the graph, ncnn
    // (whose .param carries no shapes) uses them to size its buffers and checks the output count
    // after the warm-up run.
    std::vector<int64_t> input_shape;
    size_t output_size = 0;
};

class ModelRunner {
public:
    virtual ~ModelRunner() = default;
    virtual const char *name() const = 0;
    // in: input_shape floats (NCHW). out: output_size floats, layout of the ONNX output
    // ([5,5040] row-major for the detector, [400] for the classifier). Throws std::runtime_error.
    virtual void run(const float *in, float *out) = 0;
    // Same as run(), calling before_submit() at the last moment before the GPU work is submitted (ncnn:
    // after the command buffer and the input upload are recorded; others: before run()). Used by the GPU
    // schedule so the detector reaches the GPU right when it is handed over.
    virtual void run_hooked(const float *in, float *out, const std::function<void()> &before_submit) {
        before_submit();
        run(in, out);
    }
};

// Throws std::runtime_error if the backend is not built in or the model cannot be loaded.
std::unique_ptr<ModelRunner> make_runner(const RunnerOptions &opt);
std::unique_ptr<ModelRunner> make_ort_runner(const RunnerOptions &opt);
#ifdef ROADSENSE_HAVE_NCNN
std::unique_ptr<ModelRunner> make_ncnn_runner(const RunnerOptions &opt);
#endif
bool runner_available(const std::string &backend);

}  // namespace signs
}  // namespace roadsense
