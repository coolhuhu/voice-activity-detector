// sherpa-onnx/csrc/vad-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "voice-activity-detector/vad-model-config.h"

#include <sstream>
#include <string>

#include "voice-activity-detector/macros.h"
#include "voice-activity-detector/text-utils.h"

namespace voice_activity_detector {

bool VadModelConfig::Validate() const {
  // Check Silero-VAD provider compatibility
  if (!silero_vad.model.empty()) {
    if (provider != "rknn" && EndsWith(silero_vad.model, ".rknn")) {
      SHERPA_ONNX_LOGE(
          "--provider is %s, which is not rknn, but you pass an rknn model "
          "'%s'",
          provider.c_str(), silero_vad.model.c_str());
      return false;
    }
    if (provider == "rknn" && EndsWith(silero_vad.model, ".onnx")) {
      SHERPA_ONNX_LOGE("--provider is rknn, but you pass an onnx model '%s'",
                       silero_vad.model.c_str());
      return false;
    }
    return silero_vad.Validate();
  }

  // Check Ten-VAD provider compatibility
  if (!ten_vad.model.empty()) {
    if (provider != "rknn" && EndsWith(ten_vad.model, ".rknn")) {
      SHERPA_ONNX_LOGE(
          "--provider is %s, which is not rknn, but you pass an rknn model "
          "'%s'",
          provider.c_str(), ten_vad.model.c_str());
      return false;
    }
    if (provider == "rknn" && EndsWith(ten_vad.model, ".onnx")) {
      SHERPA_ONNX_LOGE("--provider is rknn, but you pass an onnx model '%s'",
                       ten_vad.model.c_str());
      return false;
    }
    return ten_vad.Validate();
  }

  if (!firered_vad.model.empty()) {
    return firered_vad.Validate();
  }

  SHERPA_ONNX_LOGE("Please provide one VAD model.");

  return false;
}

std::string VadModelConfig::ToString() const {
  std::ostringstream os;

  os << "VadModelConfig(";
  os << "silero_vad=" << silero_vad.ToString() << ", ";
  os << "ten_vad=" << ten_vad.ToString() << ", ";
  os << "sample_rate=" << sample_rate << ", ";
  os << "num_threads=" << num_threads << ", ";
  os << "provider=\"" << provider << "\", ";
  os << "debug=" << (debug ? "True" : "False") << ")";

  return os.str();
}

}  // namespace voice_activity_detector
