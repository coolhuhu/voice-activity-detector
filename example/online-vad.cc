#include "online-vad.h"

#include <cassert>
#include <iostream>

// AcceptWaveform processes incoming audio chunks and outputs speech segments.
//
// This method implements a chunked VAD processing strategy that:
// 1. Processes audio in fixed-size windows (e.g., 512 samples for 16kHz)
// 2. Outputs speech segments progressively as they are detected
// 3. Tracks segment status (BEG, MIDDLE, END) for proper streaming
//
// Key concepts:
// - window_size: The processing unit size (e.g., 512 samples)
// - chunk_size: The input audio size (e.g., 1600 samples for 100ms at 16kHz)
// - num_processed_samples_: Tracks how many samples have been output to
// segments
// - speech_segment_start_: Indicates if we're currently in a speech segment
// - segment_id_: Unique identifier for the current speech segment
//
// The algorithm divides into two phases:
// Phase 1: Process complete windows (offset + window_size < n)
// Phase 2: Handle remaining samples (< window_size)
void OnlineVoiceActivityDetector::AcceptWaveform(const float *samples,
                                                 int32_t n) {
  if (n <= 0) {
    return;
  }

  int32_t window_size = vad_detector_.WindowSize();

  // ========================================================================
  // Phase 1: Process complete windows
  // ========================================================================
  // Process as many complete windows as possible.
  // For example, with n=1600 and window_size=512, we process 3 full windows.
  int32_t offset = 0;
  while (offset + window_size < n) {
    vad_detector_.AcceptWaveform(samples + offset, window_size);

    offset += window_size;

    // Detect the start of the speech segment
    if (!speech_segment_start_ && vad_detector_.IsSpeechDetected()) {
      speech_segment_start_ = true;
      segment_id_++;
      is_first_segment_in_current_speech_ = true;
      num_processed_samples_ = vad_detector_.CurrentSpeechSegmentStart();
    }

    if (vad_detector_.IsSpeechDetected()) {
      auto s = vad_detector_.CurrentSpeechSegment();
      int32_t tail_size = s.start + s.samples.size() - num_processed_samples_;

      if (tail_size > 0) {
        SpeechSegment segment;
        segment.start = num_processed_samples_;
        segment.segment_id = segment_id_;
        segment.samples =
            std::vector<float>(s.samples.end() - tail_size, s.samples.end());
        segment.status = is_first_segment_in_current_speech_
                             ? SpeechSegmentStatus::kBEG
                             : SpeechSegmentStatus::kMIDDLE;

        speech_segments_.push_back(std::move(segment));
        is_first_segment_in_current_speech_ = false;
        num_processed_samples_ = s.start + s.samples.size();
      }
    }

    while (!vad_detector_.Empty()) {
      auto s = vad_detector_.Front();
      vad_detector_.Pop();

      // Check if there are remaining samples in this completed segment
      // that haven't been output yet
      if (num_processed_samples_ <
          (s.start + static_cast<int32_t>(s.samples.size()))) {
        int32_t tail_chunk_size = s.start +
                                  static_cast<int32_t>(s.samples.size()) -
                                  num_processed_samples_;

        SpeechSegment segment;
        segment.start = num_processed_samples_;
        segment.segment_id = segment_id_;
        // Extract the remaining samples from this completed segment
        segment.samples = std::vector<float>(s.samples.end() - tail_chunk_size,
                                             s.samples.end());
        segment.status = SpeechSegmentStatus::kEND;

        speech_segments_.push_back(std::move(segment));
        num_processed_samples_ = s.start + s.samples.size();
      } else {
        // All samples have been output, but we still need to mark the endpoint
        // in this case, start = -1.
        SpeechSegment segment;
        segment.segment_id = segment_id_;
        segment.status = SpeechSegmentStatus::kEND;
        speech_segments_.push_back(std::move(segment));
      }

      speech_segment_start_ = false;
    }
  }

  // ========================================================================
  // Phase 2: Handle remaining samples (< window_size)
  // ========================================================================
  // After processing complete windows, there may be remaining samples that
  // don't form a full window. For example, with n=1600 and window_size=512:
  //   - Process 3 windows: 3 * 512 = 1536 samples
  //   - Remaining: 1600 - 1536 = 64 samples
  //
  // These remaining samples are fed to the detector but won't trigger
  // immediate output (they're stored in the detector's internal buffer).
  // They will be processed when more samples arrive or when Flush() is called.
  vad_detector_.AcceptWaveform(samples + offset, n - offset);

  // If speech is detected in the remaining samples, handle segment start
  // This is the same logic as in Phase 1 but for the partial window
  if (vad_detector_.IsSpeechDetected()) {
    if (!speech_segment_start_) {
      speech_segment_start_ = true;
      segment_id_++;
      is_first_segment_in_current_speech_ = true;
      num_processed_samples_ = vad_detector_.CurrentSpeechSegmentStart();
    }

    auto s = vad_detector_.CurrentSpeechSegment();
    int32_t tail_size = s.start + s.samples.size() - num_processed_samples_;

    if (tail_size > 0) {
      SpeechSegment segment;
      segment.start = num_processed_samples_;
      segment.segment_id = segment_id_;
      segment.samples =
          std::vector<float>(s.samples.end() - tail_size, s.samples.end());
      segment.status = is_first_segment_in_current_speech_
                           ? SpeechSegmentStatus::kBEG
                           : SpeechSegmentStatus::kMIDDLE;

      speech_segments_.push_back(std::move(segment));
      is_first_segment_in_current_speech_ = false;
      num_processed_samples_ = s.start + s.samples.size();
    }
  }
}

void OnlineVoiceActivityDetector::AcceptWaveform(
    const std::vector<float> &samples) {
  AcceptWaveform(samples.data(), samples.size());
}

void OnlineVoiceActivityDetector::Reset() {
  vad_detector_.Reset();
  speech_segments_.clear();
  num_processed_samples_ = 0;
  speech_segment_start_ = false;
  segment_id_ = -1;
  is_first_segment_in_current_speech_ = false;
}

bool OnlineVoiceActivityDetector::Empty() const {
  return speech_segments_.empty();
}

void OnlineVoiceActivityDetector::PopAll() { speech_segments_.clear(); }

std::vector<SpeechSegment> OnlineVoiceActivityDetector::GetSpeechSegment() {
  std::vector<SpeechSegment> segments;
  segments.reserve(speech_segments_.size());
  while (!speech_segments_.empty()) {
    segments.push_back(std::move(speech_segments_.front()));
    speech_segments_.pop_front();
  }
  return segments;
}

// Flush is called when the audio stream ends (e.g., end of file).
// It ensures that all remaining audio samples are properly processed and
// output.
//
// Why this is necessary:
// 1. The underlying VAD detector uses an internal buffer (last_) that
// accumulates
//    samples until a full window_size (e.g., 512) is available. Samples that
//    don't form a complete window remain unprocessed in this buffer.
//
// 2. During AcceptWaveform(), only complete windows are processed via
//    vad_detector_.IsSpeech(), and only segments with sufficient silence
//    duration are pushed to the output queue. If the audio ends without enough
//    trailing silence, the final speech segment remains in the detector and is
//    never output.
//
// 3. Without Flush(), the last 203 samples (for a 160459-sample file with
//    window_size=512) would be lost in the last_ buffer and never appear in any
//    output segment.
//
// What this implementation does:
// 1. Calls vad_detector_.Flush() to process any remaining samples in the
//    underlying last_ buffer and push the final speech segment to the
//    detector's output queue.
//
// 2. Checks if there is a pending speech segment in the detector's queue.
//
// 3. If present, calculates the unprocessed tail portion:
//    - The detector's segment starts at s.start and contains s.samples
//    - num_processed_samples_ tracks how many samples we've already output
//    - The tail contains samples from num_processed_samples_ to the end
//
// 4. Creates an END-status segment to properly terminate the speech segment,
//    ensuring all audio data is accounted for in the output.
void OnlineVoiceActivityDetector::Flush() {
  // Step 1: Flush the underlying VAD detector to process any remaining samples
  // in its internal last_ buffer (samples < window_size that weren't processed
  // during AcceptWaveform) and push the final speech segment to the output
  // queue.
  vad_detector_.Flush();

  // Step 2: Check if there's a pending speech segment that needs to be output.
  // This segment was likely not pushed during normal processing because the
  // audio ended before sufficient silence was detected.
  if (!vad_detector_.Empty()) {
    auto &s = vad_detector_.Front();

    // Step 3: Calculate the tail portion of the segment that hasn't been output
    // yet. s.samples contains the full speech segment from the detector.
    // num_processed_samples_ is the number of samples we've already output to
    // segments. We need to extract the portion from num_processed_samples_ to
    // the end.
    //
    // Example:
    //   - Detector segment: s.start=140768, s.samples.size()=19488 (total:
    //   160256)
    //   - num_processed_samples_: 160053 (already output)
    //   - Tail size: 160256 - 160053 = 203 samples
    //   - Extract samples[19488-203 : 19488] (last 203 samples)
    SpeechSegment segment;
    segment.start = num_processed_samples_;
    segment.samples = std::vector<float>(
        s.samples.end() - (s.start + s.samples.size() - num_processed_samples_),
        s.samples.end());
    segment.segment_id = segment_id_;
    segment.status = SpeechSegmentStatus::kEND;

    speech_segments_.push_back(std::move(segment));
  }
}