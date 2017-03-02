find_package(PkgConfig)
pkg_check_modules (PC_LIBRTLSDR librtlsdr)

find_path(
    LIBRTLSDR_INCLUDE_DIRS
    NAMES rtl-sdr.h
    HINTS ${PC_LIBRTLSDR_INCLUDE_DIRS}
    PATHS /usr/include
          /usr/local/include
)

find_library(
    LIBRTLSDR_LIBRARIES
    NAMES rtlsdr
    HINTS ${PC_LIBRTLSDR_LIBRARY_DIRS}
    PATHS /usr/lib
          /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBRTLSDR DEFAULT_MSG
                                  LIBRTLSDR_LIBRARIES LIBRTLSDR_INCLUDE_DIRS)

mark_as_advanced(LIBRTLSDR_LIBRARIES LIBRTLSDR_INCLUDE_DIRS)
