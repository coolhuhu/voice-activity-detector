// sherpa-onnx/csrc/vad-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_VAD_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_VAD_MODEL_CONFIG_H_

#include <string>

#include "voice-activity-detector/silero-vad-model-config.h"
#include "voice-activity-detector/ten-vad-model-config.h"

namespace voice_activity_detector {

struct VadModelConfig {
  SileroVadModelConfig silero_vad;
  TenVadModelConfig ten_vad;

  int32_t sample_rate = 8000;
  int32_t num_threads = 1;
  std::string provider = "cpu";

  // true to show debug information when loading models
  bool debug = false;

  VadModelConfig() = default;

  bool Validate() const;

  std::string ToString() const;
};

}  // namespace voice_activity_detector

#endif  // SHERPA_ONNX_CSRC_VAD_MODEL_CONFIG_H_
