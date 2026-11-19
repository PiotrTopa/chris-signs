# roadsense-signs: MTSD traffic-sign detector + classifier + SORT tracker (signd) and its
# offline replay tool (signd_replay), on ONNX Runtime (CPU) and optionally ncnn (Vulkan).
#
# ncnn backend (--backend ncnn): -DROADSENSE_NCNN=ON plus either
#   -DNCNN_DIR=/path/to/ncnn   an ncnn source tree built in <NCNN_DIR>/build (roadsense scripts/build-ncnn.sh)
#   (no NCNN_DIR)              an installed ncnn found by find_package(ncnn)
#
# Builds against the IPC headers of a roadsense checkout (MIT, header-only, no linking):
#   camerad/ring.h       frame-ring reader
#   camerad/gpusched.h   optional GPU turn-taking with modeld (--gpu-sync)
#   camerad/governor.h   the load governor's rate caps (signs_hz, lights_hz)
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
foreach(h ring.h jsonsock.h jpeg_nv12.h gpusched.h governor.h)
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

add_library(signd_core STATIC signd/yolo.cc signd/ort_runner.cc signd/tracker.cc signd/output.cc signd/lights.cc
  signd/verify.cc)
target_include_directories(signd_core PUBLIC ${CMAKE_SOURCE_DIR}/signd ${ROADSENSE_DIR}/camerad
  ${SIGND_ORT_INCLUDE_DIR})
target_link_libraries(signd_core PUBLIC ${SIGND_ORT_LIBRARY})

option(ROADSENSE_NCNN "Build the ncnn (Vulkan) inference backend" OFF)
set(NCNN_DIR "$ENV{NCNN_DIR}" CACHE PATH "ncnn source tree with a build in <NCNN_DIR>/build (empty: find_package(ncnn))")
if(ROADSENSE_NCNN)
  target_sources(signd_core PRIVATE signd/ncnn_runner.cc)
  target_compile_definitions(signd_core PUBLIC ROADSENSE_HAVE_NCNN)
  if(NCNN_DIR)
    set(_nb ${NCNN_DIR}/build)
    if(NOT EXISTS ${_nb}/src/libncnn.a)
      message(FATAL_ERROR "NCNN_DIR=${NCNN_DIR}: ${_nb}/src/libncnn.a not found")
    endif()
    find_package(OpenMP REQUIRED)
    target_include_directories(signd_core SYSTEM PRIVATE ${NCNN_DIR}/src ${_nb}/src)
    # Same static link group as roadsense modeld/CMakeLists.txt.
    target_link_libraries(signd_core PUBLIC
      ${_nb}/src/libncnn.a
      -Wl,--start-group
      ${_nb}/glslang/SPIRV/libSPIRV.a
      ${_nb}/glslang/glslang/libglslang.a
      ${_nb}/glslang/glslang/libMachineIndependent.a
      ${_nb}/glslang/glslang/libGenericCodeGen.a
      ${_nb}/glslang/glslang/OSDependent/Unix/libOSDependent.a
      ${_nb}/glslang/glslang/libglslang-default-resource-limits.a
      -Wl,--end-group
      OpenMP::OpenMP_CXX Threads::Threads ${CMAKE_DL_LIBS})
    message(STATUS "signd: ncnn backend from ${NCNN_DIR}")
  else()
    find_package(ncnn REQUIRED)
    target_link_libraries(signd_core PUBLIC ncnn)
    message(STATUS "signd: ncnn backend from installed ncnn (${ncnn_DIR})")
  endif()
endif()
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

add_executable(lights_replay tools/lights_replay.cc)
target_link_libraries(lights_replay PRIVATE signd_core)
if(JPEG_FOUND)
  target_compile_definitions(lights_replay PRIVATE SIGND_HAVE_LIBJPEG)
  target_include_directories(lights_replay PRIVATE ${JPEG_INCLUDE_DIRS})
  target_link_libraries(lights_replay PRIVATE ${JPEG_LIBRARIES})
endif()

# Sign tracker on precomputed detections (roadsense lane L24-S evaluation: rates, view schedules)
add_executable(signs_track_replay tools/signs_track_replay.cc)
target_link_libraries(signs_track_replay PRIVATE signd_core)

# LightTracker checks without a model: build/lights_tracker_test (also `ctest`)
add_executable(lights_tracker_test tools/lights_tracker_test.cc)
target_link_libraries(lights_tracker_test PRIVATE signd_core)
enable_testing()
add_test(NAME lights_tracker COMMAND lights_tracker_test)
add_executable(signs_tracker_test tools/signs_tracker_test.cc)
target_link_libraries(signs_tracker_test PRIVATE signd_core)
add_test(NAME signs_tracker COMMAND signs_tracker_test)
# Road users of the light detector (roadsense lane L42): rider pairing and the message key, no model needed
add_executable(road_users_test tools/road_users_test.cc)
target_link_libraries(road_users_test PRIVATE signd_core)
add_test(NAME road_users COMMAND road_users_test)

install(TARGETS signd signd_replay lights_replay RUNTIME DESTINATION bin)
