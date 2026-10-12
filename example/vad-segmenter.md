# VAD Segmenter

A command-line tool for splitting long audio files into speech segments using Voice Activity Detection (VAD).

## Overview

`vad-segmenter` processes a batch of audio files listed in a text file, detects speech segments using VAD models (Silero or Ten), and saves each segment as a separate WAV file.

## Usage

```bash
./vad-segmenter <wavlist.txt> <model.onnx> \
    --output-dir <dir> \
    --vad-type <silero|ten> \
    --sample-rate <rate> \
    --window-size <size> \
    [--threshold 0.5] \
    [--min-silence-duration 0.5] \
    [--min-speech-duration 0.25] \
    [--max-speech-duration 20] \
    [--num-threads 1] \
    [--provider cpu] \
    [--debug]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `wavlist.txt` | Text file containing one audio file path per line |
| `model.onnx` | Path to the VAD model file (.onnx) |

### Required Options

| Option | Short | Description |
|--------|-------|-------------|
| `--output-dir` | `-o` | Output directory for segmented audio files |
| `--vad-type` | `-t` | VAD model type: `silero` or `ten` |
| `--sample-rate` | | Sample rate in Hz (e.g., 8000, 16000) |
| `--window-size` | | Window size in samples |

### Optional Options

| Option | Default | Description |
|--------|---------|-------------|
| `--threshold` | 0.5 | Speech probability threshold (0.0 - 1.0) |
| `--min-silence-duration` | 0.5 | Minimum silence duration in seconds |
| `--min-speech-duration` | 0.25 | Minimum speech duration in seconds |
| `--max-speech-duration` | 20 | Maximum speech duration in seconds |
| `--num-threads` | 1 | Number of threads for ONNX runtime |
| `--provider` | cpu | ONNX runtime provider (e.g., cpu, cuda) |
| `--debug` | false | Enable debug output |

## Window Size Requirements

The window size must match the sample rate and VAD type:

| VAD Type | Sample Rate | Valid Window Sizes |
|----------|-------------|-------------------|
| Silero VAD | 16000 Hz | 512 |
| Silero VAD | 8000 Hz | 256 |
| Ten VAD | 16000 Hz | 512 |
| Ten VAD | 8000 Hz | 256 |

## Examples

### Example 1: Silero VAD with 16kHz audio

Create a file list (`wavlist.txt`):
```
/path/to/audio1.wav
/path/to/audio2.wav
/path/to/audio3.wav
```

Run the segmenter:
```bash
./vad-segmenter wavlist.txt silero_vad.onnx \
    --output-dir ./segments \
    --vad-type silero \
    --sample-rate 16000 \
    --window-size 512
```

### Example 2: Ten VAD with custom thresholds

```bash
./vad-segmenter wavlist.txt ten_vad.onnx \
    --output-dir ./output \
    --vad-type ten \
    --sample-rate 8000 \
    --window-size 256 \
    --threshold 0.6 \
    --min-silence-duration 0.3 \
    --min-speech-duration 0.5
```

## Output File Naming

For each input audio file, segments are saved with the following naming convention:

```
<input_basename>_seg_<segment_number>_<start_time>-<end_time>.wav
```

Where:
- `<segment_number>`: Sequential segment number (starting from 1)
- `<start_time>`: Segment start time in seconds (3 decimal places)
- `<end_time>`: Segment end time in seconds (3 decimal places)

For example, if the input file is `/path/to/recording.wav`, the output will be:
```
recording_seg_1_0.500-2.345.wav
recording_seg_2_3.200-5.890.wav
recording_seg_3_6.500-9.123.wav
...
```

## Output Directory

- If the output directory does not exist, it will be created automatically (including parent directories)
- If directory creation fails, an error message will be displayed and the program will exit

## wavlist.txt Format

The `wavlist.txt` file should contain one audio file path per line:

```
# Comments start with # and are ignored
/path/to/audio1.wav
/path/to/audio2.wav
/path/to/audio3.wav
```

- Empty lines are ignored
- Lines starting with `#` are treated as comments
- Leading and trailing whitespace is trimmed

