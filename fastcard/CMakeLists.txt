##
# FastCarD CMake file
#
# Options:
#   FFT_LIB: "FFTW" or "GPU_FFT" (Rpi)
#            FFT library that should be used (default: FFTW)
##

cmake_minimum_required (VERSION 2.6)
project (fastcard)
enable_language(C)

# the version number
set (FASTCARD_VERSION_MAJOR 0)
set (FASTCARD_VERSION_MINOR 1)

# use C99
add_definitions(-std=c99)

# select the release build type by default to get optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Build type not specified: defaulting to release.")
endif()

# add option to choose FFT library
set(FFT_LIB "FFTW" CACHE STRING "FFT library that should be used")
set(FFT_LIB_VALUES "FFTW;GPU_FFT")
set_property(CACHE FFT_LIB PROPERTY STRINGS ${FFT_LIB_VALUES})
message(STATUS "FFT_LIB='${FFT_LIB}'")

if (${FFT_LIB} MATCHES "FFTW")
    set (USE_FFTW ON)
    set (EXTRA_LIBS ${EXTRA_LIBS} fftw3f)
elseif (${FFT_LIB} MATCHES "GPU_FFT")
    set (USE_GPUFFT ON)
    # message(FATAL_ERROR "Work in progress")
else()
    message(FATAL_ERROR "Unknown FFT library selected: " ${FFT_LIB})
endif()

# add base64 dependency
add_library(Base64Functions lib/base64.c)
set (EXTRA_LIBS ${EXTRA_LIBS} Base64Functions)

# configure a header file to pass some of the CMake settings to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/configuration.h.in"
  "${PROJECT_BINARY_DIR}/configuration.h"
  )

# add the binary tree to the search path for include files so that we will find
# configuration.h
include_directories("${PROJECT_BINARY_DIR}")

add_executable(fastcard fastcard.c)
target_link_libraries (fastcard m volk ${EXTRA_LIBS})

