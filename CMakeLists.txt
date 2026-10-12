cmake_minimum_required(VERSION 3.14)
project(voice-activity-detector)

set(VOICE_ACTIVITY_DETECTOR_VERSION "0.3.0")

option(BUILD_SHARED_LIBS "Build shared libraries instead of static libraries" OFF)
option(ENABLE_PYTHON "Build Python bindings" OFF)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type: Debug or Release" FORCE)
  message(STATUS "CMAKE_BUILD_TYPE not set, defaulting to Release")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -DNDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
else()
  message(FATAL_ERROR "CMAKE_BUILD_TYPE should be Debug or Release")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(NOT WIN32)
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(BUILD_RPATH_USE_ORIGIN TRUE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

if(NOT APPLE)
  set(VAD_RPATH_ORIGIN "$ORIGIN")
else()
  set(VAD_RPATH_ORIGIN "@loader_path")
endif()

if(NOT WIN32)
  set(CMAKE_INSTALL_RPATH ${VAD_RPATH_ORIGIN})
  set(CMAKE_BUILD_RPATH ${VAD_RPATH_ORIGIN})
endif()

set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 17)

if(BUILD_SHARED_LIBS OR ENABLE_PYTHON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(kaldi-native-fbank)
include(eigen)

if(NOT TARGET onnxruntime)
  include(onnxruntime)
  if(TARGET onnxruntime)
    message(STATUS "TARGET onnxruntime had built.")
  endif()
endif()

if(${ENABLE_PYTHON})
  include(pybind11)
endif()

add_subdirectory(voice-activity-detector)
add_subdirectory(example)

if(${ENABLE_PYTHON})
  add_subdirectory(python)
endif()
