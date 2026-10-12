#include <iostream>

#include "online-vad.h"
#include "wav.h"

int main(int argc, char **argv) {
  const char *kUsageMessage =
      R"usage(./offline-vad-example silero_vad.onnx /path/to/input.wav)usage";

  if (argc != 3) {
    std::cerr << kUsageMessage << std::endl;
    return -1;
  }

  std::string val_model = argv[1];
  std::string input_file = argv[2];

  int32_t sample_rate = -1;
  bool is_ok = false;
  std::vector<float> samples =
      voice_activity_detector::ReadWave(input_file, &sample_rate, &is_ok);

  voice_activity_detector::VadModelConfig vad_config;
  vad_config.silero_vad.model = val_model;
  vad_config.silero_vad.threshold = 0.5;
  vad_config.silero_vad.min_silence_duration = 0.2;
  vad_config.silero_vad.min_speech_duration = 0.25;
  vad_config.silero_vad.max_speech_duration = 20;
  if (sample_rate == 16000) {
    vad_config.silero_vad.window_size = 512;  // 512 samples for 16000
  } else if (sample_rate == 8000) {
    vad_config.silero_vad.window_size = 256;  // 256 samples for 8000
  } else {
    std::cerr << "only support 8K or 16K sample rate." << std::endl;
    exit(-1);
  }

  vad_config.debug = true;

  vad_config.sample_rate = sample_rate;

  OnlineVoiceActivityDetector vad_detector(vad_config);

  vad_detector.AcceptWaveform(samples.data(), samples.size());
  vad_detector.Flush();

  if (!vad_detector.Empty()) {
    auto segments = vad_detector.GetSpeechSegment();
    vad_detector.PopAll();

    std::cout << "segments:" << std::endl;
    for (auto &s : segments) {
      if (s.start != -1) {
        std::cout << s.start << " --- " << s.start + s.samples.size()
                  << std::endl;
      }
    }
  }
}