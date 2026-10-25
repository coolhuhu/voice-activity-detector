// sherpa-onnx/csrc/voice-activity-detector-python.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "voice-activity-detector/firered-vad-model-config.h"
#include "voice-activity-detector/silero-vad-model-config.h"
#include "voice-activity-detector/ten-vad-model-config.h"
#include "voice-activity-detector/vad-model-config.h"
#include "voice-activity-detector/voice-activity-detector.h"

namespace py = pybind11;

namespace voice_activity_detector {

// 绑定 SpeechSegment
static void BindSpeechSegment(py::module &m) {
  py::class_<SpeechSegment>(m, "SpeechSegment")
      .def(py::init<>())
      .def_readwrite("start", &SpeechSegment::start)
      .def_property("samples", [](SpeechSegment &self) -> py::array_t<float> {
          return py::array_t<float>(self.samples.size(), self.samples.data());
        }, [](SpeechSegment &self, const py::array_t<float> &arr) {
          auto buf = arr.request();
          self.samples.assign(static_cast<float *>(buf.ptr), static_cast<float *>(buf.ptr) + buf.size);
        });
}

// 绑定 SileroVadModelConfig
static void BindSileroVadModelConfig(py::module &m) {
  py::class_<SileroVadModelConfig>(m, "SileroVadModelConfig")
      .def(py::init<>())
      .def_readwrite("model", &SileroVadModelConfig::model)
      .def_readwrite("threshold", &SileroVadModelConfig::threshold)
      .def_readwrite("min_silence_duration",
                     &SileroVadModelConfig::min_silence_duration)
      .def_readwrite("min_speech_duration",
                     &SileroVadModelConfig::min_speech_duration)
      .def_readwrite("window_size", &SileroVadModelConfig::window_size)
      .def_readwrite("max_speech_duration",
                     &SileroVadModelConfig::max_speech_duration)
      .def("Validate", &SileroVadModelConfig::Validate)
      .def("__str__", &SileroVadModelConfig::ToString);
}

// 绑定 TenVadModelConfig
static void BindTenVadModelConfig(py::module &m) {
  py::class_<TenVadModelConfig>(m, "TenVadModelConfig")
      .def(py::init<>())
      .def_readwrite("model", &TenVadModelConfig::model)
      .def_readwrite("threshold", &TenVadModelConfig::threshold)
      .def_readwrite("min_silence_duration",
                     &TenVadModelConfig::min_silence_duration)
      .def_readwrite("min_speech_duration",
                     &TenVadModelConfig::min_speech_duration)
      .def_readwrite("window_size", &TenVadModelConfig::window_size)
      .def_readwrite("max_speech_duration",
                     &TenVadModelConfig::max_speech_duration)
      .def("Validate", &TenVadModelConfig::Validate)
      .def("__str__", &TenVadModelConfig::ToString);
}

// Bind FireRed VAD model configuration
static void BindFireRedVadModelConfig(py::module &m) {
  py::class_<FireRedVadModelConfig>(m, "FireRedVadModelConfig")
      .def(py::init<>())
      .def_readwrite("model", &FireRedVadModelConfig::model)
      .def_readwrite("threshold", &FireRedVadModelConfig::threshold)
      .def_readwrite("min_silence_duration",
                     &FireRedVadModelConfig::min_silence_duration)
      .def_readwrite("min_speech_duration",
                     &FireRedVadModelConfig::min_speech_duration)
      .def_readwrite("window_size", &FireRedVadModelConfig::window_size)
      .def_readwrite("max_speech_duration",
                     &FireRedVadModelConfig::max_speech_duration)
      .def("Validate", &FireRedVadModelConfig::Validate)
      .def("__str__", &FireRedVadModelConfig::ToString);
}

// 绑定 VadModelConfig
static void BindVadModelConfig(py::module &m) {
  py::class_<VadModelConfig>(m, "VadModelConfig")
      .def(py::init<>())
      .def_readwrite("silero_vad", &VadModelConfig::silero_vad)
      .def_readwrite("ten_vad", &VadModelConfig::ten_vad)
      .def_readwrite("firered_vad", &VadModelConfig::firered_vad)
      .def_readwrite("sample_rate", &VadModelConfig::sample_rate)
      .def_readwrite("num_threads", &VadModelConfig::num_threads)
      .def_readwrite("provider", &VadModelConfig::provider)
      .def_readwrite("debug", &VadModelConfig::debug)
      .def("Validate", &VadModelConfig::Validate)
      .def("__str__", &VadModelConfig::ToString);
}

// 绑定 VoiceActivityDetector
static void BindVoiceActivityDetector(py::module &m) {
  py::class_<VoiceActivityDetector>(m, "VoiceActivityDetector")
      .def(py::init<const VadModelConfig &, float>(), py::arg("config"),
           py::arg("buffer_size_in_seconds") = 60)
      .def("AcceptWaveform",
           [](VoiceActivityDetector &self, const py::array_t<float> &audio) {
             self.AcceptWaveform(audio.data(), audio.size());
           })
      .def("Compute",
           [](VoiceActivityDetector &self, const py::array_t<float> &audio)
               -> float { return self.Compute(audio.data(), audio.size()); })
      .def("Empty", &VoiceActivityDetector::Empty)
      .def("Pop", &VoiceActivityDetector::Pop)
      .def("Clear", &VoiceActivityDetector::Clear)
      .def("WindowSize", &VoiceActivityDetector::WindowSize)
      .def("Front", &VoiceActivityDetector::Front,
           py::return_value_policy::reference_internal)
      .def("IsSpeechDetected", &VoiceActivityDetector::IsSpeechDetected)
      .def("CurrentSpeechSegmentSize",
           &VoiceActivityDetector::CurrentSpeechSegmentSize)
      .def("CurrentSpeechSegmentStart",
           &VoiceActivityDetector::CurrentSpeechSegmentStart)
      .def("CurrentSpeechSegment", &VoiceActivityDetector::CurrentSpeechSegment)
      .def("Reset", &VoiceActivityDetector::Reset)
      .def("Flush", &VoiceActivityDetector::Flush)
      .def("GetConfig", &VoiceActivityDetector::GetConfig,
           py::return_value_policy::reference_internal);
}

}  // namespace voice_activity_detector

PYBIND11_MODULE(_voice_activity_detector, m) {
  m.doc() = "Python wrapper for voice activity detection";

  voice_activity_detector::BindSpeechSegment(m);
  voice_activity_detector::BindSileroVadModelConfig(m);
  voice_activity_detector::BindTenVadModelConfig(m);
  voice_activity_detector::BindFireRedVadModelConfig(m);
  voice_activity_detector::BindVadModelConfig(m);
  voice_activity_detector::BindVoiceActivityDetector(m);
}
