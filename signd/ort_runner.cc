#include <onnxruntime_cxx_api.h>

#include <stdexcept>

#include "runner.h"

namespace roadsense {
namespace signs {

namespace {

std::string shape_str(const std::vector<int64_t> &s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? "," : "") + std::to_string(s[i]);
    return r + "]";
}

class OrtRunner : public ModelRunner {
public:
    explicit OrtRunner(const RunnerOptions &opt) : in_shape_(opt.input_shape), out_n_(opt.output_size) {
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(opt.threads);
        so.SetInterOpNumThreads(1);
        so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (!opt.allow_spinning) so.AddConfigEntry("session.intra_op.allow_spinning", "0");
        session_ = std::make_unique<Ort::Session>(env_, opt.model.c_str(), so);
        Ort::AllocatorWithDefaultOptions alloc;
        in_name_ = session_->GetInputNameAllocated(0, alloc).get();
        out_name_ = session_->GetOutputNameAllocated(0, alloc).get();
        const auto gi = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const auto go = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (gi != in_shape_)
            throw std::runtime_error(opt.model + ": input is " + shape_str(gi) + ", expected " + shape_str(in_shape_));
        size_t n = 1;
        for (int64_t d : go) n *= size_t(d);
        if (n != out_n_)
            throw std::runtime_error(opt.model + ": output is " + shape_str(go) + " (" + std::to_string(n) +
                                     " values), expected " + std::to_string(out_n_));
        out_shape_ = go;
        in_n_ = 1;
        for (int64_t d : in_shape_) in_n_ *= size_t(d);
    }

    const char *name() const override { return "ort"; }

    void run(const float *in, float *out) override {
        // ORT's CPU EP reads and writes the user buffers in place (no copy).
        Ort::Value iv = Ort::Value::CreateTensor<float>(mem_, const_cast<float *>(in), in_n_, in_shape_.data(), in_shape_.size());
        Ort::Value ov = Ort::Value::CreateTensor<float>(mem_, out, out_n_, out_shape_.data(), out_shape_.size());
        const char *i = in_name_.c_str(), *o = out_name_.c_str();
        session_->Run(Ort::RunOptions{nullptr}, &i, &iv, 1, &o, &ov, 1);
    }

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "signd"};
    Ort::MemoryInfo mem_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::unique_ptr<Ort::Session> session_;
    std::string in_name_, out_name_;
    std::vector<int64_t> in_shape_, out_shape_;
    size_t in_n_ = 0, out_n_ = 0;
};

}  // namespace

std::unique_ptr<ModelRunner> make_ort_runner(const RunnerOptions &opt) { return std::make_unique<OrtRunner>(opt); }

bool runner_available(const std::string &backend) {
    if (backend == "ort") return true;
#ifdef ROADSENSE_HAVE_NCNN
    if (backend == "ncnn" || backend == "ncnn-cpu") return true;
#endif
    return false;
}

std::unique_ptr<ModelRunner> make_runner(const RunnerOptions &opt) {
    if (opt.backend == "ort") return make_ort_runner(opt);
#ifdef ROADSENSE_HAVE_NCNN
    if (opt.backend == "ncnn" || opt.backend == "ncnn-cpu") return make_ncnn_runner(opt);
#endif
    throw std::runtime_error("backend '" + opt.backend + "' not available in this build (ort" +
                             (runner_available("ncnn") ? ", ncnn, ncnn-cpu)" : ")"));
}

}  // namespace signs
}  // namespace roadsense
