##
# Fastdet CMake file
#
# This is a mess, I know.
##

cmake_minimum_required (VERSION 2.6)
project (fastdet)
enable_language(CXX)

# show all errors
add_definitions(-Wall)
add_definitions(-Wextra)
add_definitions(-Wno-unused-parameter)
add_definitions(-Wno-unused)
add_definitions(-Wsign-compare)

# use c++11
add_definitions(-std=c++11)

# the version number
set(VERSION_INFO_MAJOR_VERSION 0)
set(VERSION_INFO_MINOR_VERSION 10)

# select the release build type by default to get optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Build type not specified: defaulting to release.")
endif()
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "")

# find dependencies
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
find_package(PkgConfig)
find_package(Fastcard)
find_package(Volk)

if(NOT FASTCARD_FOUND)
    message(FATAL_ERROR "Fastcard required to compile fastdet")
endif()

if(NOT VOLK_FOUND)
    message(FATAL_ERROR "volk required to compile fastdet")
endif()

include_directories(
    ${FASTCARD_INCLUDE_DIRS}
    ${VOLK_INCLUDE_DIRS}
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
    ${CMAKE_CURRENT_SOURCE_DIR}/fastdet.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/fastdet.pc
@ONLY)

# create uninstall target
configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
@ONLY)

add_custom_target(uninstall
    ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
)

# add targets
add_library(fastdet
            SHARED
            corr_detector.cpp
            fastcard_wrappers.cpp)

# FIXME: corr_detector should be independent from fastcard
target_link_libraries (fastdet
                       ${FASTCARD_LIBRARIES}
                       ${VOLK_LIBRARIES}
                       m)

add_executable(fastdet_bin fastdet.cpp)
target_link_libraries (fastdet_bin fastdet)

set_target_properties(fastdet_bin
  PROPERTIES OUTPUT_NAME fastdet)

# add install targets
install (TARGETS fastdet_bin DESTINATION bin)

install (TARGETS fastdet DESTINATION lib)

install (FILES corr_detector.h
               fastcard_wrappers.h
         DESTINATION include/fastdet)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fastdet.pc
         DESTINATION lib/pkgconfig)
