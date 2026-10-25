set(vad_srcs
  circular-buffer.cc
  file-utils.cc
  onnx-utils.cc
  text-utils.cc
  provider.cc
  provider-config.cc
  silero-vad-model.cc
  silero-vad-model-config.cc
  ten-vad-model.cc
  ten-vad-model-config.cc
  vad-model.cc
  vad-model-config.cc
  voice-activity-detector.cc
  firered-vad-model-config.cc
  firered-vad-model.cc
)

# only build shared library
add_library(voice-activity-detector SHARED ${vad_srcs})

set_target_properties(
    voice-activity-detector
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON  
)

target_include_directories(voice-activity-detector PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(voice-activity-detector PRIVATE 
  onnxruntime 
  kaldi-native-fbank-core 
  eigen
)

set(INTERFACE
  voice-activity-detector.h
  vad-model-config.h
  silero-vad-model-config.h
  ten-vad-model-config.h
  firered-vad-model-config.h
)

install(TARGETS voice-activity-detector
  DESTINATION lib
)

install(FILES ${INTERFACE}
  DESTINATION include/voice-activity-detector
)