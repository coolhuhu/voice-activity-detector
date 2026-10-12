function(download_pybind11)
  include(FetchContent)

  set(pybind11_URL  "https://github.com/pybind/pybind11/archive/refs/tags/v3.0.2.tar.gz")
  set(pybind11_HASH "SHA256=2f20a0af0b921815e0e169ea7fec63909869323581b89d7de1553468553f6a2d")

  # If you don't have access to the Internet,
  # please pre-download pybind11
  set(possible_file_locations
    ${CMAKE_SOURCE_DIR}/third-part/pybind11-3.0.2.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(pybind11_URL  "${f}")
      file(TO_CMAKE_PATH "${pybind11_URL}" pybind11_URL)
      message(STATUS "Found local downloaded pybind11: ${pybind11_URL}")
      break()
    endif()
  endforeach()

  FetchContent_Declare(pybind11
    URL
      ${pybind11_URL}
    URL_HASH          ${pybind11_HASH}
  )

  FetchContent_GetProperties(pybind11)
  if(NOT pybind11_POPULATED)
    message(STATUS "Downloading pybind11 from ${pybind11_URL}")
    FetchContent_Populate(pybind11)
  endif()
  message(STATUS "pybind11 is downloaded to ${pybind11_SOURCE_DIR}")
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR} EXCLUDE_FROM_ALL)
endfunction()

download_pybind11()
