# 语音活动检测器 (VAD)

本项目提供了开箱即用的语音活动检测（VAD）模块。

## 依赖

- **[ONNX Runtime](https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-linux-x64-1.14.0.tgz)** （目前仅在 x86_64 平台进行了测试，跨平台支持由 ONNX Runtime 提供。）

- **[Kaldi Native FBank](https://github.com/csukuangfj/kaldi-native-fbank/archive/refs/tags/v1.22.3.tar.gz)**  用于特征提取。

- **[kissfft](https://github.com/mborgerding/kissfft/archive/febd4caeed32e33ad8b2e0bb5ea77542c40f18ec.zip)** Kaldi Native FBank 依赖 kissfft。

- **[Eigen](https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz)**  用于计算的线性代数库。

- **[pybind11](https://github.com/pybind/pybind11/archive/refs/tags/v3.0.2.tar.gz)** (optional) 用于构建 python 接口。

若不能连接互联网，需将上述依赖库下载至项目根目录下的 `third-part` 目录下，具体下载链接参考 cmake 目录中对应的 cmake 脚本。

## 编译

1. 克隆本仓库。
2. 执行以下命令进行项目编译：

```sh
cd voice-activity-detector
cmake -B build
cmake --build build
```

## 目录结构

```
model/
├── silero_vad_v5.onnx
├── silero_vad_v6.onnx
├── ten-vad.onnx
└── ten-vad.int8.onnx

example/
├── silero-vad.cc          # Silero VAD 示例程序
├── ten-vad.cc             # TEN VAD 示例程序
├── online-vad.h           # 流式与非流式调用接口头文件
├── online-vad.cc          # 流式与非流式调用接口实现
├── offline-vad-example.cc # 非流式处理示例
├── online-vad-example.cc  # 流式处理示例
├── vad-segmenter.cc       # 切分语音片段的命令行程序，开箱即用
├── vad-segmenter.md       # vad-segmenter.cc 程序使用文档
├── wav.h                  # WAV 文件处理头文件
└── wav.cc                 # WAV 文件处理实现
```

## python 调用

克隆本项目

```sh
cd voice-activity-detector
# 从源码安装 python 包
pip install .

# 安装后验证
python -c "import voice_activity_detector; print(voice_activity_detector.__file__)"
```

若不能联网，请提前下载好[依赖库](#依赖)。python 接口见 [voice_activity_detector.py](python/voice_activity_detector/voice_activity_detector.py)，接口调用参考 [test_voice_activity_detector.py](python/test/test_voice_activity_detector.py)。