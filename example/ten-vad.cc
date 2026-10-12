#include <iostream>
#include <string>

#include "voice-activity-detector/voice-activity-detector.h"
#include "wav.h"

int main(int argc, char **argv) {
  const char *kUsageMessage =
      R"usage(./ten-vad ten_vad.onnx /path/to/input.wav /path/to/output.wav)usage";

  if (argc != 4) {
    std::cerr << kUsageMessage << std::endl;
    return -1;
  }

  std::string val_model = argv[1];
  std::string input_file = argv[2];
  std::string output_file = argv[3];

  voice_activity_detector::VadModelConfig vad_config;
  vad_config.ten_vad.model = val_model;
  vad_config.ten_vad.threshold = 0.5;
  vad_config.ten_vad.min_silence_duration = 0.2;
  vad_config.ten_vad.min_speech_duration = 0.25;
  vad_config.ten_vad.max_speech_duration = 20;
  vad_config.debug = true;

  int32_t sample_rate = -1;
  bool is_ok = false;
  std::vector<float> samples =
      voice_activity_detector::ReadWave(input_file, &sample_rate, &is_ok);
  if (!is_ok) {
    std::cerr << "Read " << input_file << " error!" << std::endl;
    exit(-1);
  }

  vad_config.sample_rate = sample_rate;

  if (vad_config.sample_rate == 16000) {
    vad_config.ten_vad.window_size = 512;
  } else {
    vad_config.ten_vad.window_size = 256;
  }

  voice_activity_detector::VoiceActivityDetector vad(vad_config);
  std::cout << vad_config.ToString() << std::endl;

  bool is_eof = false;
  int offset = 0;
  int window_size = vad_config.ten_vad.window_size;

  std::vector<float> samples_without_silence;

  while (!is_eof) {
    if (offset + window_size < samples.size()) {
      vad.AcceptWaveform(samples.data() + offset, window_size);
      offset += window_size;
    } else {
      is_eof = true;
      vad.Flush();
    }

    while (!vad.Empty()) {
      auto &segment = vad.Front();
      float start_time = segment.start / static_cast<float>(sample_rate);
      float end_time =
          start_time + segment.samples.size() / static_cast<float>(sample_rate);
      printf("%.3f -- %.3f\n", start_time, end_time);

      samples_without_silence.insert(samples_without_silence.end(),
                                     segment.samples.begin(),
                                     segment.samples.end());
      vad.Pop();
    }
  }

  voice_activity_detector::WriteWave(output_file, sample_rate,
                                     samples_without_silence.data(),
                                     samples_without_silence.size());
  std::cout << "Saved to " << output_file << std::endl;

  return 0;
}