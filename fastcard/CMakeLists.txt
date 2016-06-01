##
# FastCarD CMake file
#
# Options:
#
#   - CMAKE_BUILD_TYPE:
#       "Release", "Debug", "RelWithDebInfo" and "MinSizeRel".
#
#   - FFT_LIB:
#       "FFTW" or "GPU_FFT".
#       FFT library that should be used (default: FFTW or GPU_FFT on Rpi).
#
#   - TUNE_BUILD:
#       Tune build to work on this specific PC without support for earlier CPUs
#       in the architecture family.
#
#   - USE_VOLK:
#       Use vector optimized library of kernels (VOLK).
##

cmake_minimum_required (VERSION 2.6)
project (fastcard)
enable_language(C)

# show all errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")

# the version number
set (FASTCARD_VERSION_MAJOR 0)
set (FASTCARD_VERSION_MINOR 2)

# use C99 with GNU extensions
# GNU extensions are required for clock_gettime
add_definitions(-std=gnu99)

# select the release build type by default to get optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Build type not specified: defaulting to release.")
endif()

# tune build
option (TUNE_BUILD 
        "Tune build to work on this specific PC without support for earlier CPUs in the architecture family." ON) 
if (TUNE_BUILD)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mtune=native")
endif (TUNE_BUILD)

# detect raspberry pi 1
execute_process(COMMAND grep -o BCM2708 /proc/cpuinfo
    OUTPUT_VARIABLE SOC_NAME)

if ("${SOC_NAME}" STREQUAL "")
    set(RASPBERRYPI1 false)
else()
    set(RASPBERRYPI1 true)
endif()

# set special flags for RPI1
if (RASPBERRYPI1)
    message(STATUS "Detected Raspberry Pi")
    # use GPU_FFT as default for RPI1
    set(FFT_LIB_DEFAULT "GPU_FFT")
else()
    set(FFT_LIB_DEFAULT "FFTW")
endif (RASPBERRYPI1)

# add option to choose FFT library
set(FFT_LIB ${FFT_LIB_DEFAULT} CACHE STRING "FFT library that should be used")
set(FFT_LIB_VALUES "FFTW;GPU_FFT")
set_property(CACHE FFT_LIB PROPERTY STRINGS ${FFT_LIB_VALUES})
message(STATUS "FFT library that will be used: ${FFT_LIB}")

# Link to "real time" shared library for clock_gettime
set (EXTRA_LIBS ${EXTRA_LIBS} rt)

if (${FFT_LIB} MATCHES "FFTW")
    set (USE_FFTW ON)
    set (EXTRA_LIBS ${EXTRA_LIBS} fftw3f m)
elseif (${FFT_LIB} MATCHES "GPU_FFT")
    set (USE_GPUFFT ON)
    add_subdirectory(lib/gpu_fft)
    set (EXTRA_LIBS ${EXTRA_LIBS} gpu_fft)
else()
    message(FATAL_ERROR "Unknown FFT library selected: " ${FFT_LIB})
endif()

# add base64 dependency
add_library(Base64Functions lib/base64.c)
set (EXTRA_LIBS ${EXTRA_LIBS} Base64Functions)

# add volk dependency
option (USE_VOLK
    "Use vector optimized library of kernels (VOLK)" ON)
if (USE_VOLK)
    set (EXTRA_LIBS ${EXTRA_LIBS} volk)
endif (USE_VOLK)

# configure a header file to pass some of the CMake settings to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/configuration.h.in"
  "${PROJECT_BINARY_DIR}/configuration.h"
  )

# add the binary tree to the search path for include files so that we will find
# configuration.h
include_directories("${PROJECT_BINARY_DIR}")

add_executable(fastcard fastcard.c)
target_link_libraries (fastcard ${EXTRA_LIBS})

