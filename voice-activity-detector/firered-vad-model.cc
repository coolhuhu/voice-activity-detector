// sherpa-onnx/csrc/fire-red-vad-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "voice-activity-detector/firered-vad-model.h"

#include <algorithm>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kaldi-native-fbank/csrc/online-feature.h"
#include "voice-activity-detector/vad-model-config.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "voice-activity-detector/file-utils.h"
#include "voice-activity-detector/macros.h"
#include "voice-activity-detector/onnx-utils.h"

namespace voice_activity_detector {

class FbankFeatureExtractor {
 public:
  explicit FbankFeatureExtractor() {
    opts_.frame_opts.dither = 0;
    opts_.frame_opts.snip_edges = true;
    opts_.frame_opts.samp_freq = 16000;
    opts_.frame_opts.frame_shift_ms = 10;
    opts_.frame_opts.frame_length_ms = 25;
    opts_.frame_opts.remove_dc_offset = true;
    opts_.frame_opts.preemph_coeff = 0.97f;
    opts_.frame_opts.window_type = "povey";
    opts_.frame_opts.round_to_power_of_two = true;

    opts_.mel_opts.num_bins = 80;  // feature dim
    opts_.mel_opts.is_librosa = false;

    fbank_ = std::make_unique<knf::OnlineFbank>(opts_);
  }

  /**
     @param sampling_rate The sampling_rate of the input waveform. If it does
                          not equal to  config.sampling_rate, we will do
                          resampling inside.
     @param waveform Pointer to a 1-D array of size n. It must be normalized to
                     the range [-1, 1].
     @param n Number of entries in waveform
   */
  void AcceptWaveform(int32_t sampling_rate, const float *waveform,
                      int32_t n) const {
    // normalize
    std::vector<float> buf(n);
    for (int32_t i = 0; i != n; ++i) {
      buf[i] = waveform[i] * 32768;
    }
    fbank_->AcceptWaveform(sampling_rate, buf.data(), n);
  }

  /**
   * InputFinished() tells the class you won't be providing any
   * more waveform.  This will help flush out the last frame or two
   * of features, in the case where snip-edges == false; it also
   * affects the return value of IsLastFrame().
   */
  void InputFinished() const { fbank_->InputFinished(); }

  int32_t NumFramesReady() const { return fbank_->NumFramesReady(); }

  /** Note: IsLastFrame() will only ever return true if you have called
   * InputFinished() (and this frame is the last frame).
   */
  bool IsLastFrame(int32_t frame) const { return fbank_->IsLastFrame(frame); }

  /** Get n frames starting from the given frame index.
   *
   * @param frame_index  The starting frame index
   * @param n  Number of frames to get.
   * @return Return a 2-D tensor of shape (n, feature_dim).
   *         which is flattened into a 1-D vector (flattened in row major)
   */
  std::vector<float> GetFrames(int32_t frame_index, int32_t n) {
    if (frame_index + n > NumFramesReady()) {
      SHERPA_ONNX_LOGE("%d + %d > %d\n", frame_index, n, NumFramesReady());
      SHERPA_ONNX_EXIT(-1);
    }

    int32_t discard_num = frame_index - last_frame_index_;
    if (discard_num < 0) {
      SHERPA_ONNX_LOGE("last_frame_index_: %d, frame_index_: %d",
                       last_frame_index_, frame_index);
      SHERPA_ONNX_EXIT(-1);
    }

    fbank_->Pop(discard_num);

    int32_t feature_dim = FeatureDim();
    std::vector<float> features(feature_dim * n);

    float *p = features.data();

    for (int32_t i = 0; i != n; ++i) {
      const float *f = fbank_->GetFrame(i + frame_index);
      std::copy(f, f + feature_dim, p);
      p += feature_dim;
    }

    last_frame_index_ = frame_index;

    return features;
  }

  int32_t FeatureDim() const { return opts_.mel_opts.num_bins; }

  // Return a reference to the number of processed frames so far
  // before subsampling..
  // Initially, it is 0. It is always less than NumFramesReady().
  //
  // The returned reference is valid as long as this object is alive.
  int32_t &GetNumProcessedFrames() { return num_processed_frames_; }

 private:
  knf::FbankOptions opts_;
  std::unique_ptr<knf::OnlineFbank> fbank_;
  int32_t last_frame_index_ = 0;
  int32_t num_processed_frames_ = 0;
};

class FireRedVadModel::Impl {
 public:
  enum class VadState {
    kSilence = 0,
    kPossibleSpeech = 1,
    kSpeech = 2,
    kPossibleSilence = 3,
  };

  explicit Impl(const VadModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        sample_rate_(config.sample_rate) {
    auto buf = ReadFile(config.firered_vad.model);
    Init(buf.data(), buf.size());
    Validate();
  }

  template <typename Manager>
  Impl(Manager *mgr, const VadModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        sample_rate_(config.sample_rate) {
    auto buf = ReadFile(mgr, config.firered_vad.model);
    Init(buf.data(), buf.size());
    Validate();
  }

  void Reset() {
    ResetPostProcessor();
    ResetStates();
    CreateFeatureExtractor();
  }

  bool IsSpeech(const float *samples, int32_t n) {
    if (n != WindowSize()) {
      SHERPA_ONNX_LOGE("n: %d != window_size: %d", n, WindowSize());
      SHERPA_ONNX_EXIT(-1);
    }
    feat_extractor_->AcceptWaveform(16000, samples, n);

    int32_t is_speech = 0;
    while (IsReady()) {
      int32_t &num_processed_frames = feat_extractor_->GetNumProcessedFrames();
      std::vector<float> feature =
          feat_extractor_->GetFrames(num_processed_frames, 1);
      num_processed_frames += 1;
      is_speech += Process(feature);
    }

    return is_speech;
  }

  float Compute(const float *samples, int32_t n) {
    if (n != WindowSize()) {
      SHERPA_ONNX_LOGE("n: %d != window_size: %d", n, WindowSize());
      SHERPA_ONNX_EXIT(-1);
    }

    feat_extractor_->AcceptWaveform(16000, samples, n);
    while (IsReady()) {
      int32_t &num_processed_frames = feat_extractor_->GetNumProcessedFrames();
      std::vector<float> feature =
          feat_extractor_->GetFrames(num_processed_frames, 1);
      num_processed_frames += 1;
      last_prob_ = Run(feature.data(), static_cast<int32_t>(feature.size()));
    }

    return last_prob_;
  }

  bool Process(const std::vector<float> &features) {
    last_prob_ = Run(features.data(), static_cast<int32_t>(features.size()));
    return ProcessOneFrame(last_prob_);
  }

  int32_t WindowShift() const { return config_.firered_vad.window_size; }

  int32_t WindowSize() const {
    return config_.firered_vad.window_size + window_overlap_;
  }

  int32_t MinSilenceDurationSamples() const { return min_silence_samples_; }

  int32_t MinSpeechDurationSamples() const { return min_speech_samples_; }

  void SetMinSilenceDuration(float s) {
    min_silence_samples_ = sample_rate_ * s;
    min_silence_frame_ = std::max(1, static_cast<int32_t>(s * 100 + 0.5f));
  }

  void SetThreshold(float threshold) {
    config_.firered_vad.threshold = threshold;
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::make_unique<Ort::Session>(env_, model_data, model_data_length,
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below

    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    std::string model_type;
    SHERPA_ONNX_READ_META_DATA_STR_ALLOW_EMPTY(model_type, "model_type");
    if (model_type != "fire-red-vad") {
      SHERPA_ONNX_LOGE("Expect model type fire-red-vad. Given: '%s'",
                       model_type.c_str());
      SHERPA_ONNX_LOGE(
          "Please download FireRedVad from sherpa-onnx. DON'T use models from "
          "elsewhere");
      SHERPA_ONNX_EXIT(-1);
    }

    min_silence_samples_ =
        sample_rate_ * config_.firered_vad.min_silence_duration;
    min_silence_frame_ =
        std::max(1, static_cast<int32_t>(
                        config_.firered_vad.min_silence_duration * 100 + 0.5f));
    min_speech_samples_ =
        sample_rate_ * config_.firered_vad.min_speech_duration;
    min_speech_frame_ =
        std::max(8, static_cast<int32_t>(
                        config_.firered_vad.min_speech_duration * 100 + 0.5f));

    Reset();
  }

  void Validate() {
    if (sample_rate_ != 16000) {
      SHERPA_ONNX_LOGE("Expected sample rate 16000. Given: %d",
                       config_.sample_rate);
      SHERPA_ONNX_EXIT(-1);
    }

    if (config_.firered_vad.window_size != 400) {
      SHERPA_ONNX_LOGE("Expected window size 400. Given: %d",
                       config_.firered_vad.window_size);
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void ResetStates() {
    std::array<int64_t, 3> shape{1, 128, 0};

    states_.clear();

    for (int32_t i = 0; i != 8; ++i) {
      Ort::Value s = Ort::Value::CreateTensor<float>(allocator_, shape.data(),
                                                     shape.size());

      states_.push_back(std::move(s));
    }
  }

  bool IsReady() const {
    return feat_extractor_->GetNumProcessedFrames() <
           feat_extractor_->NumFramesReady();
  }

  float Run(const float *features, int32_t n) {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> x_shape = {1, 1, n};

    Ort::Value x =
        Ort::Value::CreateTensor(memory_info, const_cast<float *>(features), n,
                                 x_shape.data(), x_shape.size());

    std::vector<Ort::Value> inputs;
    inputs.reserve(input_names_.size());

    inputs.push_back(std::move(x));

    for (auto &s : states_) {
      inputs.push_back(std::move(s));
    }

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    for (int32_t i = 0; i != static_cast<int32_t>(states_.size()); ++i) {
      states_[i] = std::move(out[i + 1]);
    }

    float prob = out[0].GetTensorData<float>()[0];
    return prob;
  }

  void CreateFeatureExtractor() {
    feat_extractor_ = std::make_unique<FbankFeatureExtractor>();
  }

  void ResetPostProcessor() {
    smooth_window_.clear();
    smooth_window_sum_ = 0;
    frame_cnt_ = 0;
    state_ = VadState::kSilence;
    speech_cnt_ = 0;
    silence_cnt_ = 0;
    hit_max_speech_ = false;
    last_speech_start_frame_ = -1;
    last_speech_end_frame_ = -1;
    last_prob_ = 0;
  }

  float SmoothProb(float prob) {
    smooth_window_.push_back(prob);
    smooth_window_sum_ += prob;

    while (static_cast<int32_t>(smooth_window_.size()) > kSmoothWindowSize) {
      smooth_window_sum_ -= smooth_window_.front();
      smooth_window_.pop_front();
    }

    return smooth_window_sum_ / smooth_window_.size();
  }

  bool ProcessOneFrame(float raw_prob) {
    ++frame_cnt_;

    float smoothed_prob = SmoothProb(raw_prob);
    bool is_speech = smoothed_prob >= config_.firered_vad.threshold;

    if (hit_max_speech_) {
      last_speech_start_frame_ = frame_cnt_;
      hit_max_speech_ = false;
    }

    switch (state_) {
      case VadState::kSilence:
        if (is_speech) {
          state_ = VadState::kPossibleSpeech;
          speech_cnt_ += 1;
        } else {
          silence_cnt_ += 1;
          speech_cnt_ = 0;
        }
        break;

      case VadState::kPossibleSpeech:
        if (is_speech) {
          speech_cnt_ += 1;
          if (speech_cnt_ >= min_speech_frame_) {
            state_ = VadState::kSpeech;
            int32_t start_frame =
                std::max(1, frame_cnt_ - speech_cnt_ + 1 - kPadStartFrame);
            last_speech_start_frame_ =
                std::max(start_frame, last_speech_end_frame_ + 1);
            silence_cnt_ = 0;
          }
        } else {
          state_ = VadState::kSilence;
          silence_cnt_ = 1;
          speech_cnt_ = 0;
        }
        break;

      case VadState::kSpeech:
        speech_cnt_ += 1;
        if (is_speech) {
          silence_cnt_ = 0;
          if (speech_cnt_ >= kMaxSpeechFrame) {
            hit_max_speech_ = true;
            speech_cnt_ = 0;
            last_speech_end_frame_ = frame_cnt_;
            last_speech_start_frame_ = -1;
          }
        } else {
          state_ = VadState::kPossibleSilence;
          silence_cnt_ += 1;
        }
        break;

      case VadState::kPossibleSilence:
        speech_cnt_ += 1;
        if (is_speech) {
          state_ = VadState::kSpeech;
          silence_cnt_ = 0;
          if (speech_cnt_ >= kMaxSpeechFrame) {
            hit_max_speech_ = true;
            speech_cnt_ = 0;
            last_speech_end_frame_ = frame_cnt_;
            last_speech_start_frame_ = -1;
          }
        } else {
          silence_cnt_ += 1;
          if (silence_cnt_ >= min_silence_frame_) {
            state_ = VadState::kSilence;
            last_speech_end_frame_ = frame_cnt_;
            last_speech_start_frame_ = -1;
            speech_cnt_ = 0;
          }
        }
        break;
    }

    return state_ == VadState::kSpeech || state_ == VadState::kPossibleSilence;
  }

 private:
  static constexpr int32_t kSmoothWindowSize = 5;
  static constexpr int32_t kPadStartFrame = 5;
  static constexpr int32_t kMaxSpeechFrame = 2000;
  int32_t min_speech_frame_ = 8;

  VadModelConfig config_;

  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  std::vector<Ort::Value> states_;
  int64_t sample_rate_;
  int32_t min_silence_samples_;
  int32_t min_speech_samples_;
  int32_t min_silence_frame_ = 30;

  int32_t window_overlap_ = 0;

  std::unique_ptr<FbankFeatureExtractor> feat_extractor_;
  std::deque<float> smooth_window_;
  float smooth_window_sum_ = 0;
  int32_t frame_cnt_ = 0;
  VadState state_ = VadState::kSilence;
  int32_t speech_cnt_ = 0;
  int32_t silence_cnt_ = 0;
  bool hit_max_speech_ = false;
  int32_t last_speech_start_frame_ = -1;
  int32_t last_speech_end_frame_ = -1;
  float last_prob_ = 0;
};

FireRedVadModel::FireRedVadModel(const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
FireRedVadModel::FireRedVadModel(Manager *mgr, const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

FireRedVadModel::~FireRedVadModel() = default;

void FireRedVadModel::Reset() { return impl_->Reset(); }

bool FireRedVadModel::IsSpeech(const float *samples, int32_t n) {
  return impl_->IsSpeech(samples, n);
}

int32_t FireRedVadModel::WindowSize() const { return impl_->WindowSize(); }

int32_t FireRedVadModel::WindowShift() const { return impl_->WindowShift(); }

int32_t FireRedVadModel::MinSilenceDurationSamples() const {
  return impl_->MinSilenceDurationSamples();
}

int32_t FireRedVadModel::MinSpeechDurationSamples() const {
  return impl_->MinSpeechDurationSamples();
}

void FireRedVadModel::SetMinSilenceDuration(float s) {
  impl_->SetMinSilenceDuration(s);
}

void FireRedVadModel::SetThreshold(float threshold) {
  impl_->SetThreshold(threshold);
}

float FireRedVadModel::Compute(const float *samples, int32_t n) {
  return impl_->Compute(samples, n);
}

#if __ANDROID_API__ >= 9
template FireRedVadModel::FireRedVadModel(AAssetManager *mgr,
                                          const VadModelConfig &config);
#endif

#if __OHOS__
template FireRedVadModel::FireRedVadModel(NativeResourceManager *mgr,
                                          const VadModelConfig &config);
#endif

}  // namespace voice_activity_detector
