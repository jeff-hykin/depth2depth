// TensorRT backend: build (once, cached) and run an engine for the Depth Anything ONNX export.
// A C interface so the Rust side needs no C++ bindings; see src/tensorrt.rs.

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::fprintf(stderr, "[tensorrt] %s\n", message);
        }
    }
};

struct Engine {
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream = nullptr;
    void* input = nullptr;
    void* output = nullptr;
    int height = 0;
    int width = 0;

    ~Engine() {
        if (input) cudaFree(input);
        if (output) cudaFree(output);
        if (stream) cudaStreamDestroy(stream);
    }
};

void fail(char* error, size_t length, const std::string& message) {
    std::snprintf(error, length, "%s", message.c_str());
}

// Parse the ONNX and build an fp16 engine; empty on failure.
std::vector<char> build(Logger& logger, const void* onnx, size_t onnx_size, char* error, size_t error_length) {
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicit_batch));
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser->parse(onnx, onnx_size)) {
        fail(error, error_length, "could not parse the ONNX model");
        return {};
    }
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
    std::unique_ptr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
        fail(error, error_length, "TensorRT could not build an engine");
        return {};
    }
    const char* bytes = static_cast<const char*>(serialized->data());
    return std::vector<char>(bytes, bytes + serialized->size());
}

}  // namespace

extern "C" {

// The TensorRT version, as major*10000 + minor*100 + patch; engines only load in the version that built them.
int d2d_trt_version() {
    return getInferLibVersion();
}

// Load `prebuilt` (a serialized engine, may be null), else the engine cached at `engine_path`, else build one from
// the ONNX bytes and cache it there first. An engine that won't load (another TensorRT, another GPU) is passed over.
// Null (and `error` set) on failure.
void* d2d_trt_open(const void* onnx, size_t onnx_size, const void* prebuilt, size_t prebuilt_size, const char* engine_path,
                   char* error, size_t error_length) {
    auto engine = std::make_unique<Engine>();
    engine->runtime.reset(nvinfer1::createInferRuntime(engine->logger));
    if (prebuilt && prebuilt_size) {
        engine->engine.reset(engine->runtime->deserializeCudaEngine(prebuilt, prebuilt_size));
    }
    std::ifstream cached(engine_path, std::ios::binary);
    if (!engine->engine && cached) {
        std::vector<char> plan((std::istreambuf_iterator<char>(cached)), std::istreambuf_iterator<char>());
        engine->engine.reset(engine->runtime->deserializeCudaEngine(plan.data(), plan.size()));
    }
    if (!engine->engine) {
        std::vector<char> plan = build(engine->logger, onnx, onnx_size, error, error_length);
        if (plan.empty()) return nullptr;
        // Written aside then renamed, so a process killed mid-write leaves no half an engine behind.
        const std::string partial = std::string(engine_path) + ".part";
        std::ofstream(partial, std::ios::binary).write(plan.data(), static_cast<std::streamsize>(plan.size()));
        std::rename(partial.c_str(), engine_path);
        engine->engine.reset(engine->runtime->deserializeCudaEngine(plan.data(), plan.size()));
        if (!engine->engine) {
            fail(error, error_length, "TensorRT could not load the engine it just built");
            return nullptr;
        }
    }
    engine->context.reset(engine->engine->createExecutionContext());
    if (engine->engine->getNbIOTensors() != 2) {
        fail(error, error_length, "expected one input and one output tensor");
        return nullptr;
    }
    cudaStreamCreate(&engine->stream);
    for (int i = 0; i < 2; ++i) {
        const char* name = engine->engine->getIOTensorName(i);
        const nvinfer1::Dims dims = engine->engine->getTensorShape(name);
        size_t count = 1;
        for (int d = 0; d < dims.nbDims; ++d) count *= static_cast<size_t>(dims.d[d]);
        const bool is_input = engine->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        void** buffer = is_input ? &engine->input : &engine->output;
        if (cudaMalloc(buffer, count * sizeof(float)) != cudaSuccess) {
            fail(error, error_length, "cudaMalloc failed");
            return nullptr;
        }
        engine->context->setTensorAddress(name, *buffer);
        if (is_input) {  // NCHW
            engine->height = static_cast<int>(dims.d[2]);
            engine->width = static_cast<int>(dims.d[3]);
        }
    }
    return engine.release();
}

void d2d_trt_input_size(void* handle, int* height, int* width) {
    auto* engine = static_cast<Engine*>(handle);
    *height = engine->height;
    *width = engine->width;
}

// `input` is 3xHxW normalised floats, `output` receives HxW metres. 0 on success.
int d2d_trt_infer(void* handle, const float* input, float* output) {
    auto* engine = static_cast<Engine*>(handle);
    const size_t pixels = static_cast<size_t>(engine->height) * static_cast<size_t>(engine->width);
    if (cudaMemcpyAsync(engine->input, input, 3 * pixels * sizeof(float), cudaMemcpyHostToDevice, engine->stream) != cudaSuccess) return 1;
    if (!engine->context->enqueueV3(engine->stream)) return 2;
    if (cudaMemcpyAsync(output, engine->output, pixels * sizeof(float), cudaMemcpyDeviceToHost, engine->stream) != cudaSuccess) return 3;
    return cudaStreamSynchronize(engine->stream) == cudaSuccess ? 0 : 4;
}

void d2d_trt_close(void* handle) {
    delete static_cast<Engine*>(handle);
}

}  // extern "C"
