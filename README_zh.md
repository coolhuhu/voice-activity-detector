# Voice Activity Detector

一个便于快速集成到语音系统中的语音活动检测（VAD）组件，来源于 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)。项目提供 C++ 共享库、Python 接口和音频切分命令行工具，支持 Silero VAD、TEN VAD 和 FireRed VAD。

## 前置条件

CMake 构建配置支持以下系统和架构：

| 系统  | 架构                                  |
| ----- | ------------------------------------- |
| Linux | x86_64、aarch64                       |
| macOS | arm64、x86_64；可配置 universal2 构建 |

构建 C++ 库需要 CMake 3.14 或更新版本、支持 C++17 的编译器。安装 Python 接口还需要 Python 3.8 或更新版本及 `pip`。构建过程会依赖 ONNX Runtime 1.17.0、Kaldi Native FBank、kissfft 和 Eigen；Python接口 构建还会依赖 pybind11。CMake 会从网络获取依赖，或优先使用 [third-part](./third-part/) 中对应平台的本地压缩包。下载源及校验值见 [cmake](./cmake/)。

## C++ 安装与使用

在仓库根目录构建并安装：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix "$PWD/install"
```

安装目录中的 `include/voice-activity-detector/` 提供公共头文件，`lib/` 提供共享库及运行依赖。模型文件需随应用单独部署，在 [model](./model/) 目录下。

音频采样率必须与所选模型匹配，组件不会自动重采样。输入采样应为 `float32`，并归一化到 `[-1, 1]`。

| 模型        | 采样率         | 配置窗口大小 | 模型文件示例                    |
| ----------- | -------------- | ------------ | ------------------------------- |
| Silero VAD  | 16 kHz / 8 kHz | 512 / 256    | `model/silero_vad_v6.onnx`      |
| TEN VAD     | 16 kHz         | 512–768      | `model/ten-vad.onnx`            |
| FireRed VAD | 16 kHz         | 400          | `model/firered_stream_vad.onnx` |

下面以 FireRed VAD 为例；`samples` 是语音系统提供的 16 kHz、单声道、归一化 `float` 采样：

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

如需批量切分 WAV 文件，可直接使用编译生成的 `build/bin/vad-segmenter`。先创建 `wavlist.txt`，每行写一个音频文件路径，再运行：

```bash
./build/bin/vad-segmenter wavlist.txt model/firered_stream_vad.onnx \
  --output-dir ./segments \
  --vad-type firered \
  --sample-rate 16000 \
  --window-size 400
```

输入 WAV 的采样率需与 `--sample-rate` 一致。其他参数和模型用法见 [vad-segmenter.md](example/vad-segmenter.md)。

## Python 安装与使用

在支持的平台上，从仓库根目录安装 Python 包；`pip` 会编译 C++ 扩展，因此同样需要上述编译环境和构建依赖：

```bash
git clone https://github.com/coolhuhu/voice-activity-detector
cd voice-activity-detector
python -m pip install .
python -c "import voice_activity_detector; print(voice_activity_detector.__file__)"
```

下面的示例读取 16 kHz、单声道、16 位 PCM WAV，并使用项目自带的 FireRed 模型提取语音片段。请在仓库根目录运行，或将模型路径改为实际部署位置：

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

更多 Python 调用方式见 [接口实现](python/voice_activity_detector/voice_activity_detector.py) 和 [测试示例](python/test/test_voice_activity_detector.py)。
