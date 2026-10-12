#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "voice-activity-detector/voice-activity-detector.h"
#include "wav.h"

namespace fs = std::filesystem;

std::string FormatFloat(float value) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3) << value;
  return oss.str();
}

// VAD type enumeration
enum VadType { kSilero = 0, kTen = 2 };

// Structure to hold command line arguments
struct CommandLineArgs {
  // Positional arguments
  std::string wavlist_path;
  std::string model_path;

  // Required options
  std::string output_dir;
  VadType vad_type;
  std::string vad_type_str;
  int32_t sample_rate;
  int32_t window_size;

  // Optional options with default values
  float threshold = 0.5f;
  float min_silence_duration = 0.5f;
  float min_speech_duration = 0.25f;
  float max_speech_duration = 20.0f;
  int32_t num_threads = 1;
  std::string provider = "cpu";
  bool debug = false;

  // Flag indicating if parsing was successful
  bool is_valid = false;
};

// Print usage message
void PrintUsage(const char *program_name) {
  std::cout
      << "VAD Segmenter - Split long audio files into speech segments\n"
      << "\nUsage:\n"
      << "  " << program_name << " <wavlist.txt> <model.onnx> \\\n"
      << "    --output-dir <dir> \\\n"
      << "    --vad-type <silero|ten> \\\n"
      << "    --sample-rate <rate> \\\n"
      << "    --window-size <size> \\\n"
      << "    [--threshold 0.5] \\\n"
      << "    [--min-silence-duration 0.5] \\\n"
      << "    [--min-speech-duration 0.25] \\\n"
      << "    [--max-speech-duration 20] \\\n"
      << "    [--num-threads 1] \\\n"
      << "    [--provider cpu] \\\n"
      << "    [--debug]\n"
      << "\nArguments:\n"
      << "  wavlist.txt          Text file containing one audio file path per "
         "line\n"
      << "  model.onnx           Path to the VAD model file\n"
      << "\nRequired options:\n"
      << "  --output-dir, -o     Output directory for segmented audio files\n"
      << "  --vad-type, -t       VAD model type: silero or ten\n"
      << "  --sample-rate        Sample rate in Hz (e.g., 8000, 16000)\n"
      << "  --window-size        Window size in samples\n"
      << "\nOptional options:\n"
      << "  --threshold          Speech probability threshold (default: 0.5)\n"
      << "  --min-silence-duration Minimum silence duration in seconds "
         "(default: 0.5)\n"
      << "  --min-speech-duration  Minimum speech duration in seconds "
         "(default: 0.25)\n"
      << "  --max-speech-duration  Maximum speech duration in seconds "
         "(default: 20)\n"
      << "  --num-threads        Number of threads for ONNX runtime "
         "(default: 1)\n"
      << "  --provider           ONNX runtime provider (default: cpu)\n"
      << "  --debug              Enable debug output\n"
      << "\nWindow size requirements:\n"
      << "  Silero VAD: 512 for 16000Hz, 256 for 8000Hz\n"
      << "  Ten VAD:    512 for 16000Hz\n"
      << "\nExample:\n"
      << "  " << program_name << " wavlist.txt silero_vad.onnx \\\n"
      << "    --output-dir ./segments \\\n"
      << "    --vad-type silero \\\n"
      << "    --sample-rate 16000 \\\n"
      << "    --window-size 512\n"
      << std::endl;
}

// Validate window size against sample rate for different VAD types
bool ValidateWindowSize(VadType vad_type, int32_t sample_rate,
                        int32_t window_size) {
  // Silero VAD validation
  if (vad_type == kSilero) {
    if (sample_rate == 16000 && window_size == 512) return true;
    if (sample_rate == 8000 && window_size == 256) return true;
    std::cerr << "Error: For Silero VAD, window_size must be 512 for 16000Hz "
                 "or 256 for 8000Hz. "
              << "Got: sample_rate=" << sample_rate
              << ", window_size=" << window_size << std::endl;
    return false;
  }

  // Ten VAD validation
  if (vad_type == kTen) {
    if (sample_rate == 16000 && (window_size >= 512 && window_size <= 768)) {
      return true;
    }

    std::cerr << "Error: For Ten VAD with 16000Hz, window_size must be between "
                 "512 and 768, "
              << "Got: sample_rate=" << sample_rate
              << ", window_size=" << window_size << std::endl;
    return false;
  }

  return false;
}

// Parse command line arguments
CommandLineArgs ParseCommandLineArgs(int argc, char **argv) {
  CommandLineArgs args;

  // Check minimum number of arguments
  if (argc < 3) {
    std::cerr << "Error: Insufficient arguments" << std::endl;
    PrintUsage(argv[0]);
    return args;
  }

  // Positional arguments
  args.wavlist_path = argv[1];
  args.model_path = argv[2];

  // Parse command line arguments
  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--output-dir" || arg == "-o") {
      if (i + 1 < argc) {
        args.output_dir = argv[++i];
      } else {
        std::cerr << "Error: --output-dir requires a value" << std::endl;
        return args;
      }
    } else if (arg == "--vad-type" || arg == "-t") {
      if (i + 1 < argc) {
        args.vad_type_str = argv[++i];
      } else {
        std::cerr << "Error: --vad-type requires a value" << std::endl;
        return args;
      }
    } else if (arg == "--sample-rate") {
      if (i + 1 < argc) {
        args.sample_rate = std::stoi(argv[++i]);
      } else {
        std::cerr << "Error: --sample-rate requires a value" << std::endl;
        return args;
      }
    } else if (arg == "--window-size") {
      if (i + 1 < argc) {
        args.window_size = std::stoi(argv[++i]);
      } else {
        std::cerr << "Error: --window-size requires a value" << std::endl;
        return args;
      }
    } else if (arg == "--threshold") {
      if (i + 1 < argc) {
        args.threshold = std::stof(argv[++i]);
      } else {
        std::cerr << "Error: --threshold requires a value" << std::endl;
        return args;
      }
    } else if (arg == "--min-silence-duration") {
      if (i + 1 < argc) {
        args.min_silence_duration = std::stof(argv[++i]);
      } else {
        std::cerr << "Error: --min-silence-duration requires a value"
                  << std::endl;
        return args;
      }
    } else if (arg == "--min-speech-duration") {
      if (i + 1 < argc) {
        args.min_speech_duration = std::stof(argv[++i]);
      } else {
        std::cerr << "Error: --min-speech-duration requires a value"
                  << std::endl;
        return args;
      }
    } else if (arg == "--max-speech-duration") {
      if (i + 1 < argc) {
        args.max_speech_duration = std::stof(argv[++i]);
      } else {
        std::cerr << "Error: --max-speech-duration requires a value"
                  << std::endl;
        return args;
      }
    } else if (arg == "--num-threads") {
      if (i + 1 < argc) {
        args.num_threads = std::stoi(argv[++i]);
      } else {
        std::cerr << "Error: --num-threads requires a value" << std::endl;
        return args;
      }
    } else if (arg == "--provider") {
      if (i + 1 < argc) {
        args.provider = argv[++i];
      } else {
        std::cerr << "Error: --provider requires a value" << std::endl;
        return args;
      }
    } else if (arg == "--debug") {
      args.debug = true;
    } else {
      std::cerr << "Error: Unknown argument: " << arg << std::endl;
      PrintUsage(argv[0]);
      return args;
    }
  }

  // Validate required parameters
  if (args.output_dir.empty()) {
    std::cerr << "Error: --output-dir is required" << std::endl;
    PrintUsage(argv[0]);
    return args;
  }

  if (args.vad_type_str.empty()) {
    std::cerr << "Error: --vad-type is required" << std::endl;
    PrintUsage(argv[0]);
    return args;
  }

  if (args.sample_rate == 0) {
    std::cerr << "Error: --sample-rate is required" << std::endl;
    PrintUsage(argv[0]);
    return args;
  }

  if (args.window_size == 0) {
    std::cerr << "Error: --window-size is required" << std::endl;
    PrintUsage(argv[0]);
    return args;
  }

  // Parse VAD type string
  if (args.vad_type_str == "silero") {
    args.vad_type = kSilero;
  } else if (args.vad_type_str == "ten") {
    args.vad_type = kTen;
  } else {
    std::cerr << "Error: Invalid VAD type '" << args.vad_type_str
              << "'. Must be: silero or ten" << std::endl;
    return args;
  }

  // Validate window size
  if (!ValidateWindowSize(args.vad_type, args.sample_rate, args.window_size)) {
    return args;
  }

  args.is_valid = true;
  return args;
}

// Create VAD configuration from command line arguments
voice_activity_detector::VadModelConfig CreateVadConfig(
    const CommandLineArgs &args) {
  voice_activity_detector::VadModelConfig vad_config;
  vad_config.sample_rate = args.sample_rate;
  vad_config.num_threads = args.num_threads;
  vad_config.provider = args.provider;
  vad_config.debug = args.debug;

  if (args.vad_type == kSilero) {
    vad_config.silero_vad.model = args.model_path;
    vad_config.silero_vad.threshold = args.threshold;
    vad_config.silero_vad.min_silence_duration = args.min_silence_duration;
    vad_config.silero_vad.min_speech_duration = args.min_speech_duration;
    vad_config.silero_vad.window_size = args.window_size;
    vad_config.silero_vad.max_speech_duration = args.max_speech_duration;
  } else if (args.vad_type == kTen) {
    vad_config.ten_vad.model = args.model_path;
    vad_config.ten_vad.threshold = args.threshold;
    vad_config.ten_vad.min_silence_duration = args.min_silence_duration;
    vad_config.ten_vad.min_speech_duration = args.min_speech_duration;
    vad_config.ten_vad.window_size = args.window_size;
    vad_config.ten_vad.max_speech_duration = args.max_speech_duration;
  }

  return vad_config;
}

// Extract base filename without extension
std::string GetBaseName(const std::string &filepath) {
  fs::path path(filepath);
  return path.stem().string();
}

// Read file paths from wavlist.txt
std::vector<std::string> ReadWavList(const std::string &wavlist_path) {
  std::vector<std::string> file_list;
  std::ifstream file(wavlist_path);

  if (!file.is_open()) {
    std::cerr << "Error: Cannot open wavlist file: " << wavlist_path
              << std::endl;
    return file_list;
  }

  std::string line;
  while (std::getline(file, line)) {
    // Skip empty lines and lines starting with #
    if (!line.empty() && line[0] != '#') {
      // Trim whitespace
      size_t start = line.find_first_not_of(" \t\r\n");
      size_t end = line.find_last_not_of(" \t\r\n");
      if (start != std::string::npos) {
        file_list.push_back(line.substr(start, end - start + 1));
      }
    }
  }

  return file_list;
}

// Process a single audio file and save segments
int ProcessAudioFile(voice_activity_detector::VoiceActivityDetector *vad,
                     const std::string &input_file,
                     const std::string &output_dir, int32_t sample_rate,
                     int32_t window_size) {
  // Read audio file
  int32_t file_sample_rate = -1;
  bool is_ok = false;
  std::vector<float> samples =
      voice_activity_detector::ReadWave(input_file, &file_sample_rate, &is_ok);

  if (!is_ok) {
    std::cerr << "Failed to read" << std::endl;
    return -1;
  }

  // Validate sample rate matches
  if (file_sample_rate != sample_rate) {
    std::cerr << "Sample rate mismatch: expected " << sample_rate << ", got "
              << file_sample_rate << std::endl;
    return -1;
  }

  // Reset VAD state for this file
  vad->Reset();

  // Process audio in chunks
  bool is_eof = false;
  int offset = 0;
  int segment_count = 0;

  while (!is_eof) {
    if (offset + window_size < samples.size()) {
      vad->AcceptWaveform(samples.data() + offset, window_size);
      offset += window_size;
    } else {
      vad->AcceptWaveform(samples.data() + offset, samples.size() - offset);
      is_eof = true;
      vad->Flush();
    }

    // Process detected segments
    while (!vad->Empty()) {
      const auto &segment = vad->Front();
      float start_time = static_cast<float>(segment.start) / sample_rate;
      float end_time =
          static_cast<float>(segment.start + segment.samples.size()) /
          sample_rate;
      std::string base_name = GetBaseName(input_file);
      fs::path output_file_path =
          fs::path(output_dir) /
          (base_name + "_seg_" + std::to_string(segment_count + 1) + "_" +
           FormatFloat(start_time) + "-" + FormatFloat(end_time) + ".wav");
      std::string output_file = output_file_path.string();

      voice_activity_detector::WriteWave(output_file, sample_rate,
                                         segment.samples.data(),
                                         segment.samples.size());

      segment_count++;
      vad->Pop();
    }
  }

  std::cout << segment_count << " segment(s)" << std::endl;
  return 0;
}

int main(int argc, char **argv) {
  CommandLineArgs args = ParseCommandLineArgs(argc, argv);
  if (!args.is_valid) {
    return -1;
  }

  if (!fs::exists(args.output_dir)) {
    std::cout << "Creating output directory: " << args.output_dir << std::endl;
    std::error_code ec;
    if (!fs::create_directories(args.output_dir, ec)) {
      std::cerr << "Error: Failed to create output directory '"
                << args.output_dir << "': " << ec.message() << std::endl;
      return -1;
    }
  }

  std::vector<std::string> file_list = ReadWavList(args.wavlist_path);
  if (file_list.empty()) {
    std::cerr << "Error: No audio files found in " << args.wavlist_path
              << std::endl;
    return -1;
  }

  std::cout << "Found " << file_list.size() << " audio file(s) to process"
            << std::endl;

  voice_activity_detector::VadModelConfig vad_config = CreateVadConfig(args);
  voice_activity_detector::VoiceActivityDetector vad(vad_config);

  int success_count = 0;
  int error_count = 0;

  for (const auto &input_file : file_list) {
    std::cout << "Processing: " << input_file << " ... " << std::flush;

    int result = ProcessAudioFile(&vad, input_file, args.output_dir,
                                  args.sample_rate, args.window_size);

    if (result == 0) {
      success_count++;
    } else {
      error_count++;
    }
  }

  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "Total files: " << file_list.size() << std::endl;
  std::cout << "Successful: " << success_count << std::endl;
  std::cout << "Failed: " << error_count << std::endl;

  return (error_count > 0) ? 1 : 0;
}
