find_package(PkgConfig)
pkg_check_modules (PC_VOLK librtlsdr)

find_path(
    VOLK_INCLUDE_DIRS
    NAMES volk/volk.h
    HINTS ${PC_VOLK_INCLUDE_DIRS}
    PATHS /usr/include
          /usr/local/include
)

find_library(
    VOLK_LIBRARIES
    NAMES volk
    HINTS ${PC_VOLK_LIBRARY_DIRS}
    PATHS /usr/lib
          /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VOLK DEFAULT_MSG
                                  VOLK_LIBRARIES VOLK_INCLUDE_DIRS)

mark_as_advanced(VOLK_LIBRARIES VOLK_INCLUDE_DIRS)
