#ifndef VAD_FILE_UTILS_H_
#define VAD_FILE_UTILS_H_

#include <fstream>
#include <string>
#include <vector>

namespace voice_activity_detector {

bool FileExists(const std::string &filename);

std::vector<char> ReadFile(const std::string &filename);

}  // namespace voice_activity_detector

#endif  // FILE_UTILS_H_