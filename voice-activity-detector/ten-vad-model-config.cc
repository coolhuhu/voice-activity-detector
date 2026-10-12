// sherpa-onnx/csrc/ten-vad-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "voice-activity-detector/ten-vad-model-config.h"

#include <sstream>
#include <string>

#include "voice-activity-detector/file-utils.h"
#include "voice-activity-detector/macros.h"

namespace voice_activity_detector {

bool TenVadModelConfig::Validate() const {
  if (model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --ten-vad-model");
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("TEN vad model file '%s' does not exist", model.c_str());
    return false;
  }

  if (threshold < 0.01) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --ten-vad-threshold. Given: %f",
        threshold);
    return false;
  }

  if (threshold >= 1) {
    SHERPA_ONNX_LOGE(
        "Please use a smaller value for --ten-vad-threshold. Given: %f",
        threshold);
    return false;
  }

  if (min_silence_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --ten-vad-min-silence-duration. "
        "Given: "
        "%f",
        min_silence_duration);
    return false;
  }

  if (min_speech_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --ten-vad-min-speech-duration. "
        "Given: "
        "%f",
        min_speech_duration);
    return false;
  }

  if (max_speech_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --ten-vad-max-speech-duration. "
        "Given: "
        "%f",
        max_speech_duration);
    return false;
  }

  return true;
}

std::string TenVadModelConfig::ToString() const {
  std::ostringstream os;

  os << "TenVadModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "threshold=" << threshold << ", ";
  os << "min_silence_duration=" << min_silence_duration << ", ";
  os << "min_speech_duration=" << min_speech_duration << ", ";
  os << "max_speech_duration=" << max_speech_duration << ", ";
  os << "window_size=" << window_size << ")";

  return os.str();
}

}  // namespace voice_activity_detector