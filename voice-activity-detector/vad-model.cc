// sherpa-onnx/csrc/vad-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "voice-activity-detector/vad-model.h"

#include "voice-activity-detector/macros.h"
#include "voice-activity-detector/silero-vad-model.h"
#include "voice-activity-detector/ten-vad-model.h"

namespace voice_activity_detector {

std::unique_ptr<VadModel> VadModel::Create(const VadModelConfig &config) {
  if (!config.silero_vad.model.empty()) {
    return std::make_unique<SileroVadModel>(config);
  }

  if (!config.ten_vad.model.empty()) {
    return std::make_unique<TenVadModel>(config);
  }

  SHERPA_ONNX_LOGE("Please provide a vad model");
  return nullptr;
}

}  // namespace voice_activity_detector
