
pybind11_add_module(_voice_activity_detector voice-activity-detector-python11.cc)
target_include_directories(_voice_activity_detector PRIVATE ${CMAKE_SOURCE_DIR})

if(NOT WIN32)
  target_link_libraries(_voice_activity_detector PRIVATE "-Wl,-rpath,${VAD_RPATH_ORIGIN}")
endif()

target_link_libraries(_voice_activity_detector PRIVATE voice-activity-detector)
install(TARGETS _voice_activity_detector DESTINATION lib)