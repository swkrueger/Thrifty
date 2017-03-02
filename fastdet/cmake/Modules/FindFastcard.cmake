find_package(PkgConfig)
pkg_check_modules (PC_FASTCARD librtlsdr)

find_path(
    FASTCARD_INCLUDE_DIRS
    NAMES fastcard/fastcard.h
    HINTS ${PC_FASTCARD_INCLUDE_DIRS}
    PATHS /usr/include
          /usr/local/include
)

find_library(
    FASTCARD_LIBRARIES
    NAMES fastcard
    HINTS ${PC_FASTCARD_LIBRARY_DIRS}
    PATHS /usr/lib
          /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FASTCARD DEFAULT_MSG
                                  FASTCARD_LIBRARIES FASTCARD_INCLUDE_DIRS)

mark_as_advanced(FASTCARD_LIBRARIES FASTCARD_INCLUDE_DIRS)
