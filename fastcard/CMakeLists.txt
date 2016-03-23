cmake_minimum_required (VERSION 2.6)
project (fastcard)
enable_language(C)

# use C99
add_definitions(-std=c99)

# select the release build type by default to get optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Build type not specified: defaulting to release.")
endif()

add_library(Base64Functions lib/base64.c)

add_executable(fastcard fastcard.c)
target_link_libraries (fastcard Base64Functions fftw3f m volk)

