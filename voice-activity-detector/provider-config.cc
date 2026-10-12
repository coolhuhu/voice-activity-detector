// sherpa-onnx/csrc/provider-config.cc
//
// Copyright (c)  2024  Uniphore (Author: Manickavela)

#include "voice-activity-detector/provider-config.h"

#include <sstream>

#include "voice-activity-detector/file-utils.h"
#include "voice-activity-detector/macros.h"

namespace voice_activity_detector {

bool CudaConfig::Validate() const {
  if (cudnn_conv_algo_search < 1 || cudnn_conv_algo_search > 3) {
    SHERPA_ONNX_LOGE(
        "cudnn_conv_algo_search: '%d' is not a valid option."
        "Options : [1,3]. Check OnnxRT docs",
        cudnn_conv_algo_search);
    return false;
  }
  return true;
}

std::string CudaConfig::ToString() const {
  std::ostringstream os;

  os << "CudaConfig(";
  os << "cudnn_conv_algo_search=" << cudnn_conv_algo_search << ")";

  return os.str();
}

bool TensorrtConfig::Validate() const {
  if (trt_max_workspace_size < 0) {
    std::ostringstream os;
    os << "trt_max_workspace_size: " << trt_max_workspace_size
       << " is not valid.";
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
    return false;
  }
  if (trt_max_partition_iterations < 0) {
    SHERPA_ONNX_LOGE("trt_max_partition_iterations: %d is not valid.",
                     trt_max_partition_iterations);
    return false;
  }
  if (trt_min_subgraph_size < 0) {
    SHERPA_ONNX_LOGE("trt_min_subgraph_size: %d is not valid.",
                     trt_min_subgraph_size);
    return false;
  }

  return true;
}

std::string TensorrtConfig::ToString() const {
  std::ostringstream os;

  os << "TensorrtConfig(";
  os << "trt_max_workspace_size=" << trt_max_workspace_size << ", ";
  os << "trt_max_partition_iterations=" << trt_max_partition_iterations << ", ";
  os << "trt_min_subgraph_size=" << trt_min_subgraph_size << ", ";
  os << "trt_fp16_enable=\"" << (trt_fp16_enable ? "True" : "False") << "\", ";
  os << "trt_detailed_build_log=\""
     << (trt_detailed_build_log ? "True" : "False") << "\", ";
  os << "trt_engine_cache_enable=\""
     << (trt_engine_cache_enable ? "True" : "False") << "\", ";
  os << "trt_engine_cache_path=\"" << trt_engine_cache_path.c_str() << "\", ";
  os << "trt_timing_cache_enable=\""
     << (trt_timing_cache_enable ? "True" : "False") << "\", ";
  os << "trt_timing_cache_path=\"" << trt_timing_cache_path.c_str() << "\",";
  os << "trt_dump_subgraphs=\"" << (trt_dump_subgraphs ? "True" : "False")
     << "\" )";
  return os.str();
}

bool ProviderConfig::Validate() const {
  if (device < 0) {
    SHERPA_ONNX_LOGE("device: '%d' is invalid.", device);
    return false;
  }

  if (provider == "cuda" && !cuda_config.Validate()) {
    return false;
  }

  if (provider == "trt" && !trt_config.Validate()) {
    return false;
  }

  return true;
}

std::string ProviderConfig::ToString() const {
  std::ostringstream os;

  os << "ProviderConfig(";
  os << "device=" << device << ", ";
  os << "provider=\"" << provider << "\", ";
  os << "cuda_config=" << cuda_config.ToString() << ", ";
  os << "trt_config=" << trt_config.ToString() << ")";
  return os.str();
}

}  // namespace voice_activity_detector
