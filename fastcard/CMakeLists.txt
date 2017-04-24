##
# FastCarD CMake file
#
# Options:
#
#   - CMAKE_BUILD_TYPE:
#       "Release", "Debug", "RelWithDebInfo" and "MinSizeRel".
#
#   - TUNE_BUILD:
#       Tune build to work on this specific PC without support for earlier CPUs
#       in the architecture family.
##

cmake_minimum_required (VERSION 2.6)
project (fastcard)
enable_language(C)

# compiler setup
add_definitions(-Wall)
add_definitions(-Wextra)
add_definitions(-Wno-unused-parameter)
add_definitions(-Wno-unused)
add_definitions(-Wsign-compare)

# support for enabling switching state of bias tee programmatically
add_definitions(-DLIBRTLSDR_BIAS_TEE_SUPPORT)

# use C99
add_definitions(-std=c99)

# set version information
set(VERSION_INFO_MAJOR_VERSION 0)
set(VERSION_INFO_MINOR_VERSION 11)

# select the release build type by default to get optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Build type not specified: defaulting to release.")
endif()
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "")

# tune build
option (TUNE_BUILD 
        "Tune build to work on this specific PC without support for earlier CPUs in the architecture family." OFF) 
if (TUNE_BUILD)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mtune=native")
endif (TUNE_BUILD)

# find dependencies
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
find_package(PkgConfig)
set (USE_FFTW ON)
find_package(FFTW3f)
find_package(Volk)
find_package(Threads)
find_package(LibRTLSDR)

if(NOT FFTW3F_FOUND)
    message(FATAL_ERROR "fftw3f required to compile fastcard")
endif()
if(NOT VOLK_FOUND)
    message(FATAL_ERROR "fftw3f required to compile fastcard")
endif()
if(NOT THREADS_FOUND)
    message(FATAL_ERROR "pthreads required to compile fastcard")
endif()
if(NOT LIBRTLSDR_FOUND)
    message(FATAL_ERROR "librtlsdr required to compile fastcard")
endif()

include_directories(
    ${FFTW3F_INCLUDE_DIRS}
    ${VOLK_INCLUDE_DIRS}
    ${LIBRTLSDR_INCLUDE_DIRS}
    ${THREADS_PTHREADS_INCLUDE_DIR}
)

set (EXTRA_LIBS
     ${EXTRA_LIBS}
     ${FFTW3F_LIBRARIES}
     ${VOLK_LIBRARIES}
     ${CMAKE_THREAD_LIBS_INIT}
     ${LIBRTLSDR_LIBRARIES}
)

# configure a header file to pass some of the CMake settings to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/configuration.h.in"
  "${PROJECT_BINARY_DIR}/configuration.h"
)

# add the binary tree to the search path for include files so that we will find
# configuration.h
include_directories("${PROJECT_BINARY_DIR}")

# create PkgConfig file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/fastcard.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/fastcard.pc
@ONLY)

# create uninstall target
configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
@ONLY)

add_custom_target(uninstall
    ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
)

list(APPEND fastcard_srcs
     cardet.c
     fft.c
     rawconv.c
     parse.c
     reader.c
     raw_reader.c
     card_reader.c
     rtlsdr_reader.c
     circbuf.c
     lib/base64.c
     fastcard.c
     fargs.c  # TODO: move 'fargs.c' to "fastcard_bin" target
)

# add targets
add_library(fastcard SHARED ${fastcard_srcs})

add_executable(fastcard_bin fastcard_cli.c)
target_link_libraries (fastcard m ${EXTRA_LIBS})
target_link_libraries (fastcard_bin fastcard)

set_target_properties(fastcard_bin
    PROPERTIES OUTPUT_NAME fastcard)

# add install targets
install (TARGETS fastcard_bin
                 fastcard
         RUNTIME DESTINATION bin
         LIBRARY DESTINATION lib
         ARCHIVE DESTINATION lib)

install (FILES cardet.h
               circbuf.h
               fargs.h
               fargs_type.h
               fastcard.h
               fft.h
               rawconv.h
               reader.h
               rtlsdr_reader.h
               parse.h
               lib/base64.h
         DESTINATION include/fastcard)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fastcard.pc
         DESTINATION lib/pkgconfig)
