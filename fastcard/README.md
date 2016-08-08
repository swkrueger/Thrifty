# fastcard: fast carrier detection
## Installation
### Requirements

Fastcard uses the following libraries:

 - FFTW3
 - libvolk
 - librtlsdr (optional)

Furthermore, fastcard requires `cmake` to compile.

To install the dependencies on Ubuntu 14.04 or Debian Jessy:

    sudo apt-get install build-essential cmake libfftw3-dev libvolk-dev rtl-sdr librtlsdr-dev gnuradio-dev
    
(for some reason `/usr/include/volk/volk.h` is not located in `libvolk-dev`, but in `gnuradio-dev`).
    
On Ubuntu 16.04:

    sudo apt-get install build-essential cmake libfftw3-dev libvolk1-dev rtl-sdr librtlsdr-dev 


### Building and installing
To compile fastcard, `cd` into the `fastcard` directory and run:

    mkdir build
    cd build
    cmake ..
    make
    sudo make install
    
On Raspberry Pi 3, replace `cmake ..` with:

    cmake .. -DTUNE_BUILD=0 -DCMAKE_C_FLAGS="-mcpu=cortex-a53 -mfpu=neon-vfpv4"


### Usage
Refer to `fastcard --help`.

Examples:
 - Read samples directly from RTL-SDR (`-i rtlsdr`), output carrier detections to `rx.card` (`-o rx.card`), set carrier detection threshold to a constant of 40:

       fastcard -i rtlsdr -o rx.card -t 40

 - Read raw data from a file:

       fastcard -i data.bin -o rx.card -t 40

 - Pipe data from `rtl_sdr` instead of reading directly from the SDR using librtlsdr:

       rtl_sdr -f 443.05e6 -s 2.4e6 -g 5 | fastcard -i - -o rx.card
