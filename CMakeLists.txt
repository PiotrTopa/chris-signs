# roadsense-signs: MTSD traffic-sign detector + classifier + SORT tracker (signd) and its
# offline replay tool (signd_replay), on ONNX Runtime (CPU).
#
# Builds against the IPC headers of a roadsense checkout (MIT, header-only, no linking):
#   camerad/ring.h       frame-ring reader
#   camerad/jsonsock.h   JSON-lines Unix socket server
#   camerad/jpeg_nv12.h  NV12 -> JPEG encoder (sign crops)
#   cmake -S . -B build -DROADSENSE_DIR=/path/to/roadsense
# On a laptop with the onnxruntime-linux-x64 release tarball: -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.xx
# signd itself only needs ONNX Runtime; signd_replay decodes JPEG inputs if libjpeg is found
# (JPEG_INCLUDE_DIR / JPEG_LIBRARY can point at an unpacked -dev package), PPM/NV12 otherwise.
cmake_minimum_required(VERSION 3.16)
project(roadsense_signs CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -O3)
option(ROADSENSE_NATIVE "Tune for the build machine CPU (-mcpu=native)" ON)
include(CheckCXXCompilerFlag)
if(ROADSENSE_NATIVE)
  check_cxx_compiler_flag(-mcpu=native HAVE_MCPU_NATIVE)
  if(HAVE_MCPU_NATIVE)
    add_compile_options(-mcpu=native)
  endif()
endif()
find_package(Threads REQUIRED)

set(ROADSENSE_DIR "${CMAKE_SOURCE_DIR}/../roadsense" CACHE PATH "roadsense checkout (for camerad/ring.h, jsonsock.h, jpeg_nv12.h)")
get_filename_component(ROADSENSE_DIR "${ROADSENSE_DIR}" ABSOLUTE)
foreach(h ring.h jsonsock.h jpeg_nv12.h)
  if(NOT EXISTS "${ROADSENSE_DIR}/camerad/${h}")
    message(FATAL_ERROR "${ROADSENSE_DIR}/camerad/${h} not found (set -DROADSENSE_DIR=/path/to/roadsense)")
  endif()
endforeach()
message(STATUS "roadsense headers: ${ROADSENSE_DIR}/camerad")

set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "onnxruntime install prefix (include/, lib/)")
find_path(SIGND_ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT}/include/onnxruntime
  PATHS /usr/include/onnxruntime /usr/local/include/onnxruntime)
find_library(SIGND_ORT_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib)
if(NOT SIGND_ORT_INCLUDE_DIR OR NOT SIGND_ORT_LIBRARY)
  message(FATAL_ERROR "onnxruntime not found (set ONNXRUNTIME_ROOT)")
endif()

add_library(signd_core STATIC signd/yolo.cc signd/tracker.cc signd/output.cc)
target_include_directories(signd_core PUBLIC ${CMAKE_SOURCE_DIR}/signd ${ROADSENSE_DIR}/camerad
  ${SIGND_ORT_INCLUDE_DIR})
target_link_libraries(signd_core PUBLIC ${SIGND_ORT_LIBRARY})
if(ONNXRUNTIME_ROOT)
  set_target_properties(signd_core PROPERTIES INTERFACE_LINK_OPTIONS "-Wl,-rpath,${ONNXRUNTIME_ROOT}/lib")
endif()

add_executable(signd signd/main.cc)
target_link_libraries(signd PRIVATE signd_core Threads::Threads)

add_executable(signd_replay tools/signd_replay.cc)
target_link_libraries(signd_replay PRIVATE signd_core)
find_package(JPEG QUIET)
if(JPEG_FOUND)
  target_compile_definitions(signd_replay PRIVATE SIGND_HAVE_LIBJPEG)
  target_include_directories(signd_replay PRIVATE ${JPEG_INCLUDE_DIRS})
  target_link_libraries(signd_replay PRIVATE ${JPEG_LIBRARIES})
else()
  message(STATUS "signd_replay: libjpeg not found, JPEG input disabled (PPM/PGM/NV12 only)")
endif()

install(TARGETS signd signd_replay RUNTIME DESTINATION bin)
