include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(
  silero-vad
  silero-vad.cc
  wav.cc
)
target_link_libraries(silero-vad PRIVATE voice-activity-detector)

add_executable(
  ten-vad
  ten-vad.cc
  wav.cc
)
target_link_libraries(ten-vad PRIVATE voice-activity-detector)

add_executable(
  online-vad-example
  online-vad-example.cc
  online-vad.cc
  wav.cc
)
target_link_libraries(online-vad-example PRIVATE voice-activity-detector)

add_executable(
  offline-vad-example
  offline-vad-example.cc
  online-vad.cc
  wav.cc
)
target_link_libraries(offline-vad-example PRIVATE voice-activity-detector)

add_executable(
  vad-segmenter
  vad-segmenter.cc
  wav.cc
)
target_link_libraries(vad-segmenter voice-activity-detector)
if (NOT WIN32)
  target_link_libraries(vad-segmenter "-Wl,-rpath,${VAD_RPATH_ORIGIN}/../lib")
endif()

install(TARGETS vad-segmenter
  DESTINATION bin
)

install(
  FILES
    vad-segmenter.md
    vad-segmenter.cc
  DESTINATION bin
)