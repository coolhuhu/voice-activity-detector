set(onnxruntime_URL 
  "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz"
)
set(onnxruntime_HASH "SHA256=ee5069252f549ef94759b6b60bdf10b2dc2cd71d064a7045dd66a052f956a68b")

set(onnxruntime_possible_file_locations
  ${VOICE_ACTIVITY_DETECTOR_THIRD_PARTY_DIR}/onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz
)

foreach(f IN LISTS onnxruntime_possible_file_locations)
  if(EXISTS ${f})
    set(onnxruntime_URL ${f})
    file(TO_CMAKE_PATH "${onnxruntime_URL}" onnxruntime_URL)
    message(STATUS "Found local downloaded onnxruntime: ${onnxruntime_URL}")
    break()
  endif()
endforeach()

FetchContent_Declare(onnxruntime
  URL
    ${onnxruntime_URL}
  URL_HASH          ${onnxruntime_HASH}
)

FetchContent_GetProperties(onnxruntime)
if(NOT onnxruntime_POPULATED)
  message(STATUS "Downloading onnxruntime from ${onnxruntime_URL}")
  FetchContent_Populate(onnxruntime)
endif()
message(STATUS "onnxruntime is downloaded to ${onnxruntime_SOURCE_DIR}")

find_library(location_onnxruntime onnxruntime
  PATHS
  "${onnxruntime_SOURCE_DIR}/lib"
  NO_CMAKE_SYSTEM_PATH
)

message(STATUS "location_onnxruntime: ${location_onnxruntime}")

add_library(onnxruntime SHARED IMPORTED)

set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${location_onnxruntime}
  INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
)

file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime*")
file(COPY ${onnxruntime_lib_files} DESTINATION ${CMAKE_BINARY_DIR}/lib)
message(STATUS "onnxruntime lib files: ${onnxruntime_lib_files}")
install(FILES ${onnxruntime_lib_files} DESTINATION lib)