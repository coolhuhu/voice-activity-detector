#ifndef VAD_PROVIDER_H_
#define VAD_PROVIDER_H_

#include <string>

#include "voice-activity-detector/provider-config.h"

namespace voice_activity_detector {

// Please refer to
// https://github.com/microsoft/onnxruntime/blob/main/java/src/main/java/ai/onnxruntime/OrtProvider.java
// for a list of available providers
enum class Provider {
  kCPU = 0,       // CPUExecutionProvider
  kCUDA = 1,      // CUDAExecutionProvider
  kCoreML = 2,    // CoreMLExecutionProvider
  kXnnpack = 3,   // XnnpackExecutionProvider
  kNNAPI = 4,     // NnapiExecutionProvider
  kTRT = 5,       // TensorRTExecutionProvider
  kDirectML = 6,  // DmlExecutionProvider
};

/**
 * Convert a string to an enum.
 *
 * @param s We will convert it to lowercase before comparing.
 * @return Return an instance of Provider.
 */
Provider StringToProvider(std::string s);

}  // namespace voice_activity_detector

#endif  // VAD_PROVIDER_H_