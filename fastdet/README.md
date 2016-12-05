# fastdet: fast detector
## Installation
### Requirements

Fastdet uses the following libraries:

 - libfastcard
 - FFTW3
 - libvolk
 - librtlsdr (optional)

Furthermore, fastdet requires `cmake` to compile.


### Building and installing
To compile fastdet, `cd` into the `fastdet` directory and run:

    mkdir build
    cd build
    cmake ..
    make
    sudo make install


### Usage
Refer to `fastdet --help`.

Examples:

 - Read samples directly from RTL-SDR (`-i rtlsdr`), discard blocks of data for which the SNR of the carrier peak is less that 12, cross-correlate incoming samples with `template.tpl`, trigger a detection when the SNR of the correlation peak exceeds 14, output detections to `rx.toad`:

    fastdet -i rtlsdr -z template.tpl -t 12s -u 14s -o rx.toad

 - Write blocks of data for which a positioning signal is detected to a `.card` file (e.g. for use with `thrifty analyze_detect`):

    fastdet -i rtlsdr -o rx.toad -x rx.card

 - Read raw data from a binary file (e.g. captured using `rtl_sdr`) and output detection information without writing a `.toad` file:

    fastdet -i data.bin

 - Read samples from a `.card` file, output detections to a `.toad` file:

    fastdet --card -i rx.card -o rx.toad
