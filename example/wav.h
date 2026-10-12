// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_WAVE_WRITER_H_
#define SHERPA_ONNX_CSRC_WAVE_WRITER_H_

#include <cstdint>
#include <string>
#include <vector>

namespace voice_activity_detector {

/** Read a wave file with expected sample rate.

  @param filename Path to a wave file. It MUST be single channel, 16-bit
                  PCM encoded.
  @param sampling_rate  On return, it contains the sampling rate of the file.
  @param is_ok On return it is true if the reading succeeded; false otherwise.

  @return Return wave samples normalized to the range [-1, 1).
*/
std::vector<float> ReadWave(const std::string &filename, int32_t *sampling_rate,
                            bool *is_ok);

std::vector<float> ReadWave(std::istream &is, int32_t *sampling_rate,
                            bool *is_ok);

std::vector<std::vector<float>> ReadWaveMultiChannel(std::istream &is,
                                                     int32_t *sampling_rate,
                                                     bool *is_ok);

std::vector<std::vector<float>> ReadWaveMultiChannel(
    const std::string &filename, int32_t *sampling_rate, bool *is_ok);

// Write a single channel wave file.
// Note that the input samples are in the range [-1, 1]. It will be multiplied
// by 32767 and saved in int16_t format in the wave file.
//
// @param filename Path to save the samples.
// @param sampling_rate Sample rate of the samples.
// @param samples Pointer to the samples
// @param n Number of samples
// @return Return true if the write succeeds; return false otherwise.
bool WriteWave(const std::string &filename, int32_t sampling_rate,
               const float *samples, int32_t n);

void WriteWave(char *buffer, int32_t sampling_rate, const float *samples,
               int32_t n);

bool WriteWave(const std::string &filename, int32_t sampling_rate,
               const float *samples_ch0, const float *samples_ch1, int32_t n);

void WriteWave(char *buffer, int32_t sampling_rate, const float *samples_ch0,
               const float *samples_ch1, int32_t n);

int64_t WaveFileSize(int32_t n_samples, int32_t num_channels = 1);

}  // namespace voice_activity_detector

#endif  // SHERPA_ONNX_CSRC_WAVE_WRITER_H_
