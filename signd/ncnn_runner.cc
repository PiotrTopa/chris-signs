// ncnn + Vulkan runner (see runner.h). Models come from tools/ncnn/make_ncnn.sh: input blob in0,
// outputs out0, out1, ... (the detector has out0 = boxes [4,5040] and out1 = scores [1,5040] because
// ncnn's final Concat of the YOLOv8 head gives wrong results, see tools/ncnn/fix_ncnn.py); they
// are extracted in order and concatenated into the ONNX output layout.
//
// Precision and the turnip workaround are the same as roadsense modeld/ncnn_backend.cc: fp16-packed
// storage with fp32 arithmetic (the Adreno 630 turnip driver has no VK_KHR_16bit_storage), and
// TU_DEBUG=flushall added before the Vulkan instance is created, without which about half the
// runs on that driver return corrupted outputs.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

#include <gpu.h>
#include <net.h>

#include "runner.h"

namespace roadsense {
namespace signs {

namespace {

std::mutex g_gpu_mu;
int g_gpu_users = 0;  // the detector and the classifier share ncnn's global Vulkan instance

class NcnnRunner : public ModelRunner {
public:
    explicit NcnnRunner(const RunnerOptions &opt) : out_n_(opt.output_size) {
        if (opt.input_shape.size() != 4 || opt.input_shape[0] != 1)
            throw std::runtime_error("ncnn: input_shape must be [1,C,H,W]");
        {
            std::lock_guard<std::mutex> lk(g_gpu_mu);
            if (g_gpu_users == 0 && opt.ncnn_flushall) {
                const char *t = getenv("TU_DEBUG");
                if (!t || !*t) setenv("TU_DEBUG", "flushall", 1);
                else if (!strstr(t, "flushall")) setenv("TU_DEBUG", (std::string(t) + ",flushall").c_str(), 1);
            }
            if (ncnn::get_gpu_count() <= 0) throw std::runtime_error("ncnn: no Vulkan device");
            ++g_gpu_users;
        }
        const bool fp16 = opt.ncnn_fp16;
        net_.opt.use_vulkan_compute = true;
        net_.opt.use_fp16_packed = fp16;
        net_.opt.use_fp16_storage = fp16;
        net_.opt.use_fp16_arithmetic = fp16;
        net_.opt.use_bf16_storage = false;
        net_.opt.use_packing_layout = true;
        net_.opt.num_threads = opt.threads > 0 ? opt.threads : 1;

        const std::string ext = ".param";
        if (opt.model.size() <= ext.size() || opt.model.compare(opt.model.size() - ext.size(), ext.size(), ext) != 0)
            throw std::runtime_error("ncnn: model must be the .param file (weights: same stem .bin): " + opt.model);
        const std::string bin = opt.model.substr(0, opt.model.size() - ext.size()) + ".bin";
        if (net_.load_param(opt.model.c_str()) != 0) throw std::runtime_error("ncnn: cannot load " + opt.model);
        if (net_.load_model(bin.c_str()) != 0) throw std::runtime_error("ncnn: cannot load " + bin);
        const auto &in = net_.input_names();
        if (in.size() != 1 || std::string(in[0]) != "in0") throw std::runtime_error("ncnn: expected one input 'in0' in " + opt.model);
        for (const char *o : net_.output_names()) {
            std::string s = o;
            if (s.compare(0, 3, "out") == 0) outs_.push_back(s);
        }
        std::sort(outs_.begin(), outs_.end());
        if (outs_.empty()) throw std::runtime_error("ncnn: no out* blobs in " + opt.model);

        c_ = int(opt.input_shape[1]);
        h_ = int(opt.input_shape[2]);
        w_ = int(opt.input_shape[3]);
        in_.create(w_, h_, c_);
        in_.fill(0.0f);
        outv_.resize(outs_.size());

        // Warm-up: builds the pipelines and uploads the weights (0.1-2 s), and tells us the output sizes.
        const auto t0 = std::chrono::steady_clock::now();
        const size_t n = extract();
        if (n != out_n_)
            throw std::runtime_error("ncnn: " + opt.model + " outputs " + std::to_string(n) + " values, expected " +
                                     std::to_string(out_n_));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const ncnn::GpuInfo &gi = ncnn::get_gpu_info(ncnn::get_default_gpu_index());
        const char *td = getenv("TU_DEBUG");
        fprintf(stderr, "signd: ncnn %s on %s (fp16 packed %d, storage %d, arithmetic %d), TU_DEBUG=%s, %zu output blob(s), warm-up %.0f ms\n",
                opt.model.c_str(), gi.device_name(), fp16 && gi.support_fp16_packed(), fp16 && gi.support_fp16_storage(),
                fp16 && gi.support_fp16_arithmetic(), td ? td : "", outs_.size(), ms);
    }

    ~NcnnRunner() override {
        net_.clear();
        std::lock_guard<std::mutex> lk(g_gpu_mu);
        if (--g_gpu_users == 0) ncnn::destroy_gpu_instance();
    }

    const char *name() const override { return "ncnn"; }

    void run(const float *in, float *out) override {
        const size_t plane = size_t(w_) * h_;
        for (int c = 0; c < c_; ++c)  // channel stride may be padded: copy per channel
            std::memcpy(in_.channel(c), in + c * plane, plane * sizeof(float));
        if (extract() != out_n_) throw std::runtime_error("ncnn: output size changed");
        size_t off = 0;
        for (const ncnn::Mat &m : outv_) {
            // Mats are unpacked fp32 with w contiguous per row; rows may be padded only across
            // channels, and ours are at most 2-D, so a flat copy per row is exact.
            if (m.dims <= 1) {
                std::memcpy(out + off, m.data, m.total() * sizeof(float));
            } else {
                for (int y = 0; y < m.h * std::max(1, m.c) * std::max(1, m.d); ++y)
                    std::memcpy(out + off + size_t(y) * m.w, (const float *)m.data + size_t(y) * m.w, size_t(m.w) * sizeof(float));
            }
            off += m.total();
        }
    }

private:
    size_t extract() {
        ncnn::Extractor ex = net_.create_extractor();
        if (ex.input("in0", in_) != 0) throw std::runtime_error("ncnn: bad input in0");
        size_t n = 0;
        for (size_t i = 0; i < outs_.size(); ++i) {
            const int r = ex.extract(outs_[i].c_str(), outv_[i]);
            if (r != 0 || outv_[i].empty() || outv_[i].elemsize != 4)
                throw std::runtime_error("ncnn: extract " + outs_[i] + " failed (" + std::to_string(r) + ")");
            if (outv_[i].dims == 3 && outv_[i].c > 1 && outv_[i].cstep != size_t(outv_[i].w) * outv_[i].h)
                throw std::runtime_error("ncnn: padded 3-D output not supported");
            n += outv_[i].total();
        }
        return n;
    }

    ncnn::Net net_;
    ncnn::Mat in_;
    std::vector<std::string> outs_;
    std::vector<ncnn::Mat> outv_;
    int c_ = 0, h_ = 0, w_ = 0;
    size_t out_n_ = 0;
};

}  // namespace

std::unique_ptr<ModelRunner> make_ncnn_runner(const RunnerOptions &opt) { return std::make_unique<NcnnRunner>(opt); }

}  // namespace signs
}  // namespace roadsense
