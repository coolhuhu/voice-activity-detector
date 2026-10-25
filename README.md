# Voice Activity Detector

[简体中文](README_zh.md)

A voice activity detection (VAD) component designed for quick integration into speech systems, derived from [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). The project provides a C++ shared library, Python bindings, and a command-line audio segmenter. It supports Silero VAD, TEN VAD, and FireRed VAD.

## Prerequisites

The CMake build configuration supports the following systems and architectures:

| System | Architectures                                            |
| ------ | -------------------------------------------------------- |
| Linux  | x86_64, aarch64                                          |
| macOS  | arm64, x86_64; a universal2 build can also be configured |

Building the C++ library requires CMake 3.14 or newer and a compiler with C++17 support. Installing the Python bindings also requires Python 3.8 or newer and `pip`. The build depends on ONNX Runtime 1.17.0, Kaldi Native FBank, kissfft, and Eigen; building the Python bindings also requires pybind11. CMake downloads dependencies when needed, or uses the platform-specific archives in [third-part](./third-part/) if available. See [cmake](./cmake/) for download sources and checksums.

## C++ Installation and Usage

Build and install from the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix "$PWD/install"
```

The installation's `include/voice-activity-detector/` directory contains the public headers, while `lib/` contains the shared library and runtime dependencies. Model files are in the [model](./model/) directory and must be deployed separately with your application.

The audio sample rate must match the selected model; the component does not resample audio. Input samples should be `float32` values normalized to `[-1, 1]`.

| Model       | Sample rate    | Window size | Example model file              |
| ----------- | -------------- | ----------- | ------------------------------- |
| Silero VAD  | 16 kHz / 8 kHz | 512 / 256   | `model/silero_vad_v6.onnx`      |
| TEN VAD     | 16 kHz         | 512–768     | `model/ten-vad.onnx`            |
| FireRed VAD | 16 kHz         | 400         | `model/firered_stream_vad.onnx` |

The following example uses FireRed VAD. `samples` contains 16 kHz, mono, normalized `float` samples supplied by your speech system:

```cpp
#include <algorithm>
#include <iostream>
#include <vector>

#include "voice-activity-detector/voice-activity-detector.h"

void DetectSpeech(const std::vector<float>& samples) {
  voice_activity_detector::VadModelConfig config;
  config.sample_rate = 16000;
  config.firered_vad.model = "model/firered_stream_vad.onnx";

  voice_activity_detector::VoiceActivityDetector vad(config);
  auto drain_segments = [&]() {
    while (!vad.Empty()) {
      const auto& segment = vad.Front();
      std::cout << "start=" << segment.start
                << ", samples=" << segment.samples.size() << '\n';
      vad.Pop();
    }
  };

  for (std::size_t offset = 0; offset < samples.size(); offset += 400) {
    const int count = std::min<std::size_t>(400, samples.size() - offset);
    vad.AcceptWaveform(samples.data() + offset, count);
    drain_segments();
  }
  vad.Flush();
  drain_segments();
}
```

To segment WAV files in batches, use the compiled `build/bin/vad-segmenter` tool. First, create `wavlist.txt` with one audio file path per line, then run:

```bash
./build/bin/vad-segmenter wavlist.txt model/firered_stream_vad.onnx \
  --output-dir ./segments \
  --vad-type firered \
  --sample-rate 16000 \
  --window-size 400
```

The input WAV sample rate must match `--sample-rate`. See [vad-segmenter.md](example/vad-segmenter.md) for other options and models.

## Python Installation and Usage

Install the Python package from the repository root on a supported platform. `pip` builds the C++ extension, so the compiler and build dependencies listed above are also required:

```bash
git clone https://github.com/coolhuhu/voice-activity-detector
cd voice-activity-detector
python -m pip install .
python -c "import voice_activity_detector; print(voice_activity_detector.__file__)"
```

The example below reads a 16 kHz, mono, 16-bit PCM WAV file and extracts speech segments using the included FireRed model. Run it from the repository root, or change the model path to its deployed location:

```python
import wave

import numpy as np
from voice_activity_detector import VoiceActivityDetector

with wave.open("input.wav", "rb") as wav:
    assert wav.getframerate() == 16000
    assert wav.getnchannels() == 1
    assert wav.getsampwidth() == 2
    audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0

vad = VoiceActivityDetector.init_firered(
    model="model/firered_stream_vad.onnx", sample_rate=16000
)


def collect_segments():
    while not vad.is_empty():
        segment = vad.front()
        print(f"{segment.start / 16000:.3f}s, {len(segment.samples)} samples")
        vad.pop()

for offset in range(0, len(audio), vad.window_size()):
    vad.accept_waveform(audio[offset:offset + vad.window_size()])
    collect_segments()

vad.flush()
collect_segments()
```

For more Python usage examples, see the [API implementation](python/voice_activity_detector/voice_activity_detector.py) and [tests](python/test/test_voice_activity_detector.py).
