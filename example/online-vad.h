#ifndef ONLINE_VAD_H_
#define ONLINE_VAD_H_

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "voice-activity-detector/voice-activity-detector.h"

enum class SpeechSegmentStatus {
  kBEG = 0,
  kMIDDLE = 1,
  kEND = 2,
  kINVALID = 3
};

inline std::string ToString(const SpeechSegmentStatus &status) {
  switch (status) {
    case SpeechSegmentStatus::kBEG:
      return "BEG";
    case SpeechSegmentStatus::kMIDDLE:
      return "MIDDLE";
    case SpeechSegmentStatus::kEND:
      return "END";
    case SpeechSegmentStatus::kINVALID:
      return "INVALID";
    default:
      return "UNKNOWN";
  }
  return "UNKNOWN";
}

struct SpeechSegment {
  int32_t start;  // in samples
  std::vector<float> samples;
  int32_t segment_id;
  SpeechSegmentStatus status;

  SpeechSegment()
      : start(-1), segment_id(0), status(SpeechSegmentStatus::kINVALID) {}
};

class OnlineVoiceActivityDetector {
 public:
  OnlineVoiceActivityDetector(voice_activity_detector::VadModelConfig config)
      : vad_detector_(config),
        num_processed_samples_(0),
        speech_segment_start_(false),
        segment_id_(-1),
        is_first_segment_in_current_speech_(false) {}

  void AcceptWaveform(const float *samples, int32_t n);
  void AcceptWaveform(const std::vector<float> &samples);

  bool Empty() const;

  void PopAll();

  void Flush();

  std::vector<SpeechSegment> GetSpeechSegment();

  void Reset();

 private:
  voice_activity_detector::VoiceActivityDetector vad_detector_;
  int32_t num_processed_samples_;
  bool speech_segment_start_;
  int32_t segment_id_;
  bool is_first_segment_in_current_speech_;
  std::deque<SpeechSegment> speech_segments_;
};

#endif  // ONLINE_VAD_H_