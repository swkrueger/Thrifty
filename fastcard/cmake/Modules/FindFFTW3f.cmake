# http://gnuradio.org/redmine/projects/gnuradio/repository/revisions/c2c8a53bee19e820859d01d8657821127c75b762/entry/cmake/Modules/FindFFTW3f.cmake
# Find single-precision (float) version of FFTW3

find_package(PkgConfig)
pkg_check_modules(PC_FFTW3F "fftw3f >= 3.0")

find_path(
    FFTW3F_INCLUDE_DIRS
    NAMES fftw3.h
    HINTS $ENV{FFTW3_DIR}/include
        ${PC_FFTW3F_INCLUDE_DIR}
    PATHS /usr/local/include
          /usr/include
)

find_library(
    FFTW3F_LIBRARIES
    NAMES fftw3f libfftw3f
    HINTS $ENV{FFTW3_DIR}/lib
        ${PC_FFTW3F_LIBDIR}
    PATHS /usr/local/lib
          /usr/lib
          /usr/lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3F DEFAULT_MSG
                                  FFTW3F_LIBRARIES FFTW3F_INCLUDE_DIRS)
mark_as_advanced(FFTW3F_LIBRARIES FFTW3F_INCLUDE_DIRS)
