#include "voice-activity-detector/file-utils.h"

namespace voice_activity_detector {

bool FileExists(const std::string &filename) {
  return std::ifstream(filename).good();
}

std::vector<char> ReadFile(const std::string &filename) {
  std::ifstream input(filename, std::ios::binary);
  std::vector<char> buffer(std::istreambuf_iterator<char>(input), {});
  return buffer;
}

}  // namespace voice_activity_detector
