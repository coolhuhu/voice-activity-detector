#include "voice-activity-detector/onnx-utils.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "voice-activity-detector/macros.h"
#include "voice-activity-detector/provider.h"

namespace voice_activity_detector {

static void OrtStatusFailure(OrtStatus *status, const char *s) {
  const auto &api = Ort::GetApi();
  const char *msg = api.GetErrorMessage(status);
  SHERPA_ONNX_LOGE(
      "Failed to enable TensorRT : %s."
      "Available providers: %s. Fallback to cuda",
      msg, s);
  api.ReleaseStatus(status);
}

Ort::SessionOptions GetSessionOptionsImpl(
    int32_t num_threads, const std::string &provider_str,
    const ProviderConfig *provider_config /*= nullptr*/) {
  Provider p = StringToProvider(provider_str);

  Ort::SessionOptions sess_opts;
  sess_opts.SetIntraOpNumThreads(num_threads);

  sess_opts.SetInterOpNumThreads(num_threads);

  std::vector<std::string> available_providers = Ort::GetAvailableProviders();
  std::ostringstream os;
  for (const auto &ep : available_providers) {
    os << ep << ", ";
  }

  // Other possible options
  // sess_opts.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
  // sess_opts.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);
  // sess_opts.EnableProfiling("profile");

  switch (p) {
    case Provider::kCPU:
      break;  // nothing to do for the CPU provider
    case Provider::kXnnpack: {
#if ORT_API_VERSION >= 12
      if (std::find(available_providers.begin(), available_providers.end(),
                    "XnnpackExecutionProvider") != available_providers.end()) {
        sess_opts.AppendExecutionProvider("XNNPACK");
      } else {
        SHERPA_ONNX_LOGE("Available providers: %s. Fallback to cpu!",
                         os.str().c_str());
      }
#else
      SHERPA_ONNX_LOGE(
          "Does not support xnnpack for onnxruntime: %d. Fallback to cpu!",
          static_cast<int32_t>(ORT_API_VERSION));
#endif
      break;
    }
    case Provider::kTRT: {
      if (provider_config == nullptr) {
        SHERPA_ONNX_LOGE(
            "Tensorrt support for Online models ony,"
            "Must be extended for offline and others");
        exit(1);
      }
      auto trt_config = provider_config->trt_config;
      struct TrtPairs {
        const char *op_keys;
        const char *op_values;
      };

      auto device_id = std::to_string(provider_config->device);
      auto trt_max_workspace_size =
          std::to_string(trt_config.trt_max_workspace_size);
      auto trt_max_partition_iterations =
          std::to_string(trt_config.trt_max_partition_iterations);
      auto trt_min_subgraph_size =
          std::to_string(trt_config.trt_min_subgraph_size);
      auto trt_fp16_enable = std::to_string(trt_config.trt_fp16_enable);
      auto trt_detailed_build_log =
          std::to_string(trt_config.trt_detailed_build_log);
      auto trt_engine_cache_enable =
          std::to_string(trt_config.trt_engine_cache_enable);
      auto trt_timing_cache_enable =
          std::to_string(trt_config.trt_timing_cache_enable);
      auto trt_dump_subgraphs = std::to_string(trt_config.trt_dump_subgraphs);
      std::vector<TrtPairs> trt_options = {
          {"device_id", device_id.c_str()},
          {"trt_max_workspace_size", trt_max_workspace_size.c_str()},
          {"trt_max_partition_iterations",
           trt_max_partition_iterations.c_str()},
          {"trt_min_subgraph_size", trt_min_subgraph_size.c_str()},
          {"trt_fp16_enable", trt_fp16_enable.c_str()},
          {"trt_detailed_build_log", trt_detailed_build_log.c_str()},
          {"trt_engine_cache_enable", trt_engine_cache_enable.c_str()},
          {"trt_engine_cache_path", trt_config.trt_engine_cache_path.c_str()},
          {"trt_timing_cache_enable", trt_timing_cache_enable.c_str()},
          {"trt_timing_cache_path", trt_config.trt_timing_cache_path.c_str()},
          {"trt_dump_subgraphs", trt_dump_subgraphs.c_str()}};
      // ToDo : Trt configs
      // "trt_int8_enable"
      // "trt_int8_use_native_calibration_table"

      std::vector<const char *> option_keys, option_values;
      for (const TrtPairs &pair : trt_options) {
        option_keys.emplace_back(pair.op_keys);
        option_values.emplace_back(pair.op_values);
      }

      std::vector<std::string> available_providers =
          Ort::GetAvailableProviders();
      if (std::find(available_providers.begin(), available_providers.end(),
                    "TensorrtExecutionProvider") != available_providers.end()) {
        const auto &api = Ort::GetApi();

        OrtTensorRTProviderOptionsV2 *tensorrt_options = nullptr;
        OrtStatus *statusC =
            api.CreateTensorRTProviderOptions(&tensorrt_options);
        OrtStatus *statusU = api.UpdateTensorRTProviderOptions(
            tensorrt_options, option_keys.data(), option_values.data(),
            option_keys.size());
        sess_opts.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);

        if (statusC) {
          OrtStatusFailure(statusC, os.str().c_str());
        }
        if (statusU) {
          OrtStatusFailure(statusU, os.str().c_str());
        }

        api.ReleaseTensorRTProviderOptions(tensorrt_options);
      }
      // break; is omitted here intentionally so that
      // if TRT not available, CUDA will be used
    }
    case Provider::kCUDA: {
      if (std::find(available_providers.begin(), available_providers.end(),
                    "CUDAExecutionProvider") != available_providers.end()) {
        // The CUDA provider is available, proceed with setting the options
        OrtCUDAProviderOptions options;

        if (provider_config != nullptr) {
          options.device_id = provider_config->device;
          options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch(
              provider_config->cuda_config.cudnn_conv_algo_search);
        } else {
          options.device_id = 0;
          // Default OrtCudnnConvAlgoSearchExhaustive is extremely slow
          options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
          // set more options on need
        }
        sess_opts.AppendExecutionProvider_CUDA(options);
      } else {
        SHERPA_ONNX_LOGE(
            "Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON. Available "
            "providers: %s. Fallback to cpu!",
            os.str().c_str());
      }
      break;
    }
    case Provider::kDirectML: {
#if defined(_WIN32) && SHERPA_ONNX_ENABLE_DIRECTML == 1
      sess_opts.DisableMemPattern();
      sess_opts.SetExecutionMode(ORT_SEQUENTIAL);
      int32_t device_id = 0;
      OrtStatus *status =
          OrtSessionOptionsAppendExecutionProvider_DML(sess_opts, device_id);
      if (status) {
        const auto &api = Ort::GetApi();
        const char *msg = api.GetErrorMessage(status);
        SHERPA_ONNX_LOGE("Failed to enable DirectML: %s. Fallback to cpu", msg);
        api.ReleaseStatus(status);
      }
#else
      SHERPA_ONNX_LOGE("DirectML is for Windows only. Fallback to cpu!");
#endif
      break;
    }
    case Provider::kCoreML: {
#if defined(__APPLE__)
      uint32_t coreml_flags = 0;
      (void)OrtSessionOptionsAppendExecutionProvider_CoreML(sess_opts,
                                                            coreml_flags);
#else
      SHERPA_ONNX_LOGE("CoreML is for Apple only. Fallback to cpu!");
#endif
      break;
    }
    case Provider::kNNAPI: {
#if __ANDROID_API__ >= 27
      SHERPA_ONNX_LOGE("Current API level %d ", (int32_t)__ANDROID_API__);

      // Please see
      // https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html#usage
      // to enable different flags
      uint32_t nnapi_flags = 0;
      // nnapi_flags |= NNAPI_FLAG_USE_FP16;
      // nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;
      OrtStatus *status = OrtSessionOptionsAppendExecutionProvider_Nnapi(
          sess_opts, nnapi_flags);

      if (status) {
        const auto &api = Ort::GetApi();
        const char *msg = api.GetErrorMessage(status);
        SHERPA_ONNX_LOGE(
            "Failed to enable NNAPI: %s. Available providers: %s. Fallback to "
            "cpu",
            msg, os.str().c_str());
        api.ReleaseStatus(status);
      } else {
        SHERPA_ONNX_LOGE("Use nnapi");
      }
#elif defined(__ANDROID_API__)
      SHERPA_ONNX_LOGE(
          "Android NNAPI requires API level >= 27. Current API level %d "
          "Fallback to cpu!",
          (int32_t)__ANDROID_API__);
#else
      SHERPA_ONNX_LOGE("NNAPI is for Android only. Fallback to cpu");
#endif
      break;
    }
  }
  return sess_opts;
}

static std::string GetInputName(Ort::Session *sess, size_t index,
                                OrtAllocator *allocator) {
// Note(fangjun): We only tested 1.17.1 and 1.11.0
// For other versions, we may need to change it
#if ORT_API_VERSION >= 12
  auto v = sess->GetInputNameAllocated(index, allocator);
  return v.get();
#else
  auto v = sess->GetInputName(index, allocator);
  std::string ans = v;
  allocator->Free(allocator, v);
  return ans;
#endif
}

static std::string GetOutputName(Ort::Session *sess, size_t index,
                                 OrtAllocator *allocator) {
// Note(fangjun): We only tested 1.17.1 and 1.11.0
// For other versions, we may need to change it
#if ORT_API_VERSION >= 12
  auto v = sess->GetOutputNameAllocated(index, allocator);
  return v.get();
#else
  auto v = sess->GetOutputName(index, allocator);
  std::string ans = v;
  allocator->Free(allocator, v);
  return ans;
#endif
}

void GetInputNames(Ort::Session *sess, std::vector<std::string> *input_names,
                   std::vector<const char *> *input_names_ptr) {
  Ort::AllocatorWithDefaultOptions allocator;
  size_t node_count = sess->GetInputCount();
  input_names->resize(node_count);
  input_names_ptr->resize(node_count);
  for (size_t i = 0; i != node_count; ++i) {
    (*input_names)[i] = GetInputName(sess, i, allocator);
    (*input_names_ptr)[i] = (*input_names)[i].c_str();
  }
}

void GetOutputNames(Ort::Session *sess, std::vector<std::string> *output_names,
                    std::vector<const char *> *output_names_ptr) {
  Ort::AllocatorWithDefaultOptions allocator;
  size_t node_count = sess->GetOutputCount();
  output_names->resize(node_count);
  output_names_ptr->resize(node_count);
  for (size_t i = 0; i != node_count; ++i) {
    (*output_names)[i] = GetOutputName(sess, i, allocator);
    (*output_names_ptr)[i] = (*output_names)[i].c_str();
  }
}

Ort::Value GetEncoderOutFrame(OrtAllocator *allocator, Ort::Value *encoder_out,
                              int32_t t) {
  std::vector<int64_t> encoder_out_shape =
      encoder_out->GetTensorTypeAndShapeInfo().GetShape();

  auto batch_size = encoder_out_shape[0];
  auto num_frames = encoder_out_shape[1];
  assert(t < num_frames);

  auto encoder_out_dim = encoder_out_shape[2];

  auto offset = num_frames * encoder_out_dim;

  std::array<int64_t, 2> shape{batch_size, encoder_out_dim};

  Ort::Value ans =
      Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());

  float *dst = ans.GetTensorMutableData<float>();
  const float *src = encoder_out->GetTensorData<float>();

  for (int32_t i = 0; i != batch_size; ++i) {
    std::copy(src + t * encoder_out_dim, src + (t + 1) * encoder_out_dim, dst);
    src += offset;
    dst += encoder_out_dim;
  }
  return ans;
}

std::string LookupCustomModelMetaData(const Ort::ModelMetadata &meta_data,
                                      const char *key,
                                      OrtAllocator *allocator) {
// Note(fangjun): We only tested 1.17.1 and 1.11.0
// For other versions, we may need to change it
#if ORT_API_VERSION >= 12
  auto v = meta_data.LookupCustomMetadataMapAllocated(key, allocator);
  return v ? v.get() : "";
#else
  auto v = meta_data.LookupCustomMetadataMap(key, allocator);
  std::string ans = v ? v : "";
  allocator->Free(allocator, v);
  return ans;
#endif
}

void PrintModelMetadata(std::ostream &os, const Ort::ModelMetadata &meta_data) {
  Ort::AllocatorWithDefaultOptions allocator;
#if ORT_API_VERSION >= 12
  std::vector<Ort::AllocatedStringPtr> v =
      meta_data.GetCustomMetadataMapKeysAllocated(allocator);
  for (const auto &key : v) {
    auto p = meta_data.LookupCustomMetadataMapAllocated(key.get(), allocator);
    os << key.get() << "=" << p.get() << "\n";
  }
#else
  int64_t num_keys = 0;
  char **keys = meta_data.GetCustomMetadataMapKeys(allocator, num_keys);
  for (int32_t i = 0; i < num_keys; ++i) {
    auto v = LookupCustomModelMetaData(meta_data, keys[i], allocator);
    os << keys[i] << "=" << v << "\n";
    allocator.Free(keys[i]);
  }

  allocator.Free(keys);
#endif
}

Ort::Value Clone(OrtAllocator *allocator, const Ort::Value *v) {
  auto type_and_shape = v->GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = type_and_shape.GetShape();

  switch (type_and_shape.GetElementType()) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
      Ort::Value ans = Ort::Value::CreateTensor<int32_t>(
          allocator, shape.data(), shape.size());
      const int32_t *start = v->GetTensorData<int32_t>();
      const int32_t *end = start + type_and_shape.GetElementCount();
      int32_t *dst = ans.GetTensorMutableData<int32_t>();
      std::copy(start, end, dst);
      return ans;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      Ort::Value ans = Ort::Value::CreateTensor<int64_t>(
          allocator, shape.data(), shape.size());
      const int64_t *start = v->GetTensorData<int64_t>();
      const int64_t *end = start + type_and_shape.GetElementCount();
      int64_t *dst = ans.GetTensorMutableData<int64_t>();
      std::copy(start, end, dst);
      return ans;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      Ort::Value ans = Ort::Value::CreateTensor<float>(allocator, shape.data(),
                                                       shape.size());
      const float *start = v->GetTensorData<float>();
      const float *end = start + type_and_shape.GetElementCount();
      float *dst = ans.GetTensorMutableData<float>();
      std::copy(start, end, dst);
      return ans;
    }
    default:
      fprintf(stderr, "Unsupported type: %d\n",
              static_cast<int32_t>(type_and_shape.GetElementType()));
      exit(-1);
      // unreachable code
      return Ort::Value{nullptr};
  }
}

Ort::Value View(Ort::Value *v) {
  auto type_and_shape = v->GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = type_and_shape.GetShape();

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  switch (type_and_shape.GetElementType()) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return Ort::Value::CreateTensor(
          memory_info, v->GetTensorMutableData<int32_t>(),
          type_and_shape.GetElementCount(), shape.data(), shape.size());
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return Ort::Value::CreateTensor(
          memory_info, v->GetTensorMutableData<int64_t>(),
          type_and_shape.GetElementCount(), shape.data(), shape.size());
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return Ort::Value::CreateTensor(
          memory_info, v->GetTensorMutableData<float>(),
          type_and_shape.GetElementCount(), shape.data(), shape.size());
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return Ort::Value::CreateTensor(
          memory_info, v->GetTensorMutableData<bool>(),
          type_and_shape.GetElementCount(), shape.data(), shape.size());
    default:
      fprintf(stderr, "Unsupported type: %d\n",
              static_cast<int32_t>(type_and_shape.GetElementType()));
      exit(-1);
      // unreachable code
      return Ort::Value{nullptr};
  }
}

float ComputeSum(const Ort::Value *v, int32_t n /*= -1*/) {
  std::vector<int64_t> shape = v->GetTensorTypeAndShapeInfo().GetShape();
  auto size = static_cast<int32_t>(
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()));
  if (n != -1 && n < size && n > 0) {
    size = n;
  }

  const float *p = v->GetTensorData<float>();

  return std::accumulate(p, p + size, 1.0f);
}

float ComputeMean(const Ort::Value *v, int32_t n /*= -1*/) {
  std::vector<int64_t> shape = v->GetTensorTypeAndShapeInfo().GetShape();
  auto size = static_cast<int32_t>(
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()));

  if (n != -1 && n < size && n > 0) {
    size = n;
  }

  auto sum = ComputeSum(v, n);
  return sum / size;
}

void PrintShape(const Ort::Value *v) {
  std::vector<int64_t> shape = v->GetTensorTypeAndShapeInfo().GetShape();
  std::ostringstream os;
  for (auto i : shape) {
    os << i << ", ";
  }
  os << "\n";
  fprintf(stderr, "%s", os.str().c_str());
}

template <typename T /*= float*/>
void Print1D(const Ort::Value *v) {
  std::vector<int64_t> shape = v->GetTensorTypeAndShapeInfo().GetShape();
  const T *d = v->GetTensorData<T>();
  std::ostringstream os;
  for (int32_t i = 0; i != static_cast<int32_t>(shape[0]); ++i) {
    os << d[i] << " ";
  }
  os << "\n";
  fprintf(stderr, "%s\n", os.str().c_str());
}

template void Print1D<int64_t>(const Ort::Value *v);
template void Print1D<int32_t>(const Ort::Value *v);
template void Print1D<float>(const Ort::Value *v);

template <typename T /*= float*/>
void Print2D(const Ort::Value *v) {
  std::vector<int64_t> shape = v->GetTensorTypeAndShapeInfo().GetShape();
  const T *d = v->GetTensorData<T>();

  std::ostringstream os;
  for (int32_t r = 0; r != static_cast<int32_t>(shape[0]); ++r) {
    for (int32_t c = 0; c != static_cast<int32_t>(shape[1]); ++c, ++d) {
      os << *d << " ";
    }
    os << "\n";
  }
  fprintf(stderr, "%s\n", os.str().c_str());
}

template void Print2D<int64_t>(const Ort::Value *v);
template void Print2D<float>(const Ort::Value *v);

void Print3D(const Ort::Value *v) {
  std::vector<int64_t> shape = v->GetTensorTypeAndShapeInfo().GetShape();
  const float *d = v->GetTensorData<float>();

  for (int32_t p = 0; p != static_cast<int32_t>(shape[0]); ++p) {
    fprintf(stderr, "---plane %d---\n", p);
    for (int32_t r = 0; r != static_cast<int32_t>(shape[1]); ++r) {
      for (int32_t c = 0; c != static_cast<int32_t>(shape[2]); ++c, ++d) {
        fprintf(stderr, "%.3f ", *d);
      }
      fprintf(stderr, "\n");
    }
  }
  fprintf(stderr, "\n");
}

void Print4D(const Ort::Value *v) {
  std::vector<int64_t> shape = v->GetTensorTypeAndShapeInfo().GetShape();
  const float *d = v->GetTensorData<float>();

  for (int32_t p = 0; p != static_cast<int32_t>(shape[0]); ++p) {
    fprintf(stderr, "---plane %d---\n", p);
    for (int32_t q = 0; q != static_cast<int32_t>(shape[1]); ++q) {
      fprintf(stderr, "---subplane %d---\n", q);
      for (int32_t r = 0; r != static_cast<int32_t>(shape[2]); ++r) {
        for (int32_t c = 0; c != static_cast<int32_t>(shape[3]); ++c, ++d) {
          fprintf(stderr, "%.3f ", *d);
        }
        fprintf(stderr, "\n");
      }
      fprintf(stderr, "\n");
    }
  }
  fprintf(stderr, "\n");
}

Ort::Value Repeat(OrtAllocator *allocator, Ort::Value *cur_encoder_out,
                  const std::vector<int32_t> &hyps_num_split) {
  std::vector<int64_t> cur_encoder_out_shape =
      cur_encoder_out->GetTensorTypeAndShapeInfo().GetShape();

  std::array<int64_t, 2> ans_shape{hyps_num_split.back(),
                                   cur_encoder_out_shape[1]};

  Ort::Value ans = Ort::Value::CreateTensor<float>(allocator, ans_shape.data(),
                                                   ans_shape.size());

  const float *src = cur_encoder_out->GetTensorData<float>();
  float *dst = ans.GetTensorMutableData<float>();
  int32_t batch_size = static_cast<int32_t>(hyps_num_split.size()) - 1;
  for (int32_t b = 0; b != batch_size; ++b) {
    int32_t cur_stream_hyps_num = hyps_num_split[b + 1] - hyps_num_split[b];
    for (int32_t i = 0; i != cur_stream_hyps_num; ++i) {
      std::copy(src, src + cur_encoder_out_shape[1], dst);
      dst += cur_encoder_out_shape[1];
    }
    src += cur_encoder_out_shape[1];
  }
  return ans;
}

CopyableOrtValue::CopyableOrtValue(const CopyableOrtValue &other) {
  *this = other;
}

CopyableOrtValue &CopyableOrtValue::operator=(const CopyableOrtValue &other) {
  if (this == &other) {
    return *this;
  }
  if (other.value) {
    Ort::AllocatorWithDefaultOptions allocator;
    value = Clone(allocator, &other.value);
  }
  return *this;
}

CopyableOrtValue::CopyableOrtValue(CopyableOrtValue &&other) noexcept {
  *this = std::move(other);
}

CopyableOrtValue &CopyableOrtValue::operator=(
    CopyableOrtValue &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  value = std::move(other.value);
  return *this;
}

std::vector<CopyableOrtValue> Convert(std::vector<Ort::Value> values) {
  std::vector<CopyableOrtValue> ans;
  ans.reserve(values.size());

  for (auto &v : values) {
    ans.emplace_back(std::move(v));
  }

  return ans;
}

std::vector<Ort::Value> Convert(std::vector<CopyableOrtValue> values) {
  std::vector<Ort::Value> ans;
  ans.reserve(values.size());

  for (auto &v : values) {
    ans.emplace_back(std::move(v.value));
  }

  return ans;
}

}  // namespace voice_activity_detector
