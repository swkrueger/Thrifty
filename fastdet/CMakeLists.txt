##
# Fastdet CMake file
#
# This is a mess, I know.
##

cmake_minimum_required (VERSION 2.6)
project (fastdet)
enable_language(CXX)

# show all errors
# -fgnu89-inline: workaround for linking to argp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")

# the version number
set (FASTDET_VERSION_MAJOR 0)
set (FASTDET_VERSION_MINOR 10)

# use ++11
add_definitions(-std=c++11)

# select the release build type by default to get optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Build type not specified: defaulting to release.")
endif()

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

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../fastcard/cmake/Modules)
find_package(Threads)
find_package(LibRTLSDR)
include_directories(${LIBRTLSDR_INCLUDE_DIRS})

include_directories("../fastcard")
find_library(FASTCARD NAMES fastcard
    PATHS
    /usr/lib
    /usr/local/lib
    ../fastcard/build  # FIXME!!!
  )
find_library(FASTCARD_BASE64 NAMES Base64Functions
    PATHS
    /usr/lib
    /usr/local/lib
    ../fastcard/build  # FIXME!!!
  )
find_library(FASTCARD_MODULES NAMES fastcard_modules
    PATHS
    /usr/lib
    /usr/local/lib
    ../fastcard/build  # FIXME!!!
  )

add_library(corr_detector
            corr_detector.cpp
            fastcard_wrappers.cpp)

# FIXME: corr_detector should be independent from fastcard
target_link_libraries (corr_detector
                       ${FASTCARD}
                       ${FASTCARD_MODULES}
                       ${FASTCARD_BASE64}
                       ${CMAKE_THREAD_LIBS_INIT}
                       ${LIBRTLSDR_LIBRARIES}
                       volk
                       fftw3f
                       m)

add_executable(fastdet fastdet.cpp)
target_link_libraries (fastdet corr_detector)

# add install targets
install (TARGETS fastdet DESTINATION bin)
