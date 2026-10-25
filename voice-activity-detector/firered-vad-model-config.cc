// sherpa-onnx/csrc/fire-red-vad-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "voice-activity-detector/firered-vad-model-config.h"

#include <sstream>
#include <string>

#include "voice-activity-detector/file-utils.h"
#include "voice-activity-detector/macros.h"

namespace voice_activity_detector {

bool FireRedVadModelConfig::Validate() const {
  if (model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --fire-red-vad-model");
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("FireRed vad model file '%s' does not exist",
                     model.c_str());
    return false;
  }

  if (threshold < 0.01) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --fire-red-vad-threshold. Given: %f",
        threshold);
    return false;
  }

  if (threshold >= 1) {
    SHERPA_ONNX_LOGE(
        "Please use a smaller value for --fire-red-vad-threshold. Given: %f",
        threshold);
    return false;
  }

  if (min_silence_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --fire-red-vad-min-silence-duration. "
        "Given: "
        "%f",
        min_silence_duration);
    return false;
  }

  if (min_speech_duration < 0.08) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --fire-red-vad-min-speech-duration. "
        "Should be greater than or equal to 0.08. "
        "Given: "
        "%f",
        min_speech_duration);
    return false;
  }

  if (max_speech_duration <= 0) {
    SHERPA_ONNX_LOGE(
        "Please use a larger value for --fire-red-vad-max-speech-duration. "
        "Given: "
        "%f",
        max_speech_duration);
    return false;
  }

  if (window_size != 400) {
    SHERPA_ONNX_LOGE("Expected window size 400. Given: %d", window_size);
    return false;
  }

  return true;
}

std::string FireRedVadModelConfig::ToString() const {
  std::ostringstream os;

  os << "FireRedVadModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "threshold=" << threshold << ", ";
  os << "min_silence_duration=" << min_silence_duration << ", ";
  os << "min_speech_duration=" << min_speech_duration << ", ";
  os << "max_speech_duration=" << max_speech_duration << ", ";
  os << "window_size=" << window_size << ")";

  return os.str();
}

}  // namespace voice_activity_detector