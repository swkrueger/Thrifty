// FastCarD: Fast Carrier Detection

#include <endian.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex>

#define FFTLIB FFTW
#if FFTLIB == FFTW
#include <fftw3.h>
#else
#error "Unknown FFT library"
#endif

typedef std::complex<float> fc_complex;

// Settings
int block_size_log2N = 13; // 8196
int history_size = 2085;

// Internal
int block_size = 1<<block_size_log2N;

// Buffers
uint16_t *raw_samples;
fc_complex *samples;
fc_complex *fft;
fc_complex lut[0x10000];

void generate_lut() {
    // generate lookup table for raw-to-complex conversion
    for (size_t i = 0; i <= 0xffff; ++i) {
#if __BYTE_ORDER == __LITTLE_ENDIAN
        lut[i] = fc_complex((float(i & 0xff) - 127.4f) * (1.0f/128.0f),
                            (float(i >> 8) - 127.4f) * (1.0f/128.0f));
#elif __BYTE_ORDER == __BIG_ENDIAN
        lut[i] = fc_complex((float(i >> 8) - 127.4f) * (1.0f/128.0f),
                            (float(i & 0xff) - 127.4f) * (1.0f/128.0f));
#else
#error "Could not determine endianness"
#endif
    }
}

void init_fft();
void free_fft();

void init_buffers() {
    raw_samples = (uint16_t*) calloc(block_size, sizeof(uint16_t));

    if (raw_samples == NULL) {
        fprintf(stderr, "init buffers failed\n");
        exit(1);
    }

    init_fft();

    generate_lut();
}

void free_buffers() {
    free_fft();
    free(raw_samples);
}

bool read_next_block(FILE *f) {
    // copy history
    size_t b = block_size - history_size;
    memcpy(raw_samples,
           raw_samples + b,
           history_size * 2);

    // read new data
    size_t c = fread(raw_samples + history_size, 2, b, f);

    if (c != b) {
        if (!feof(f)) {
            perror("Short read");
        }
        return false;
    }
    return true;
}

void convert_raw_to_complex() {
    for (int i = 0; i < block_size; ++i) {
        samples[i] = lut[raw_samples[i]];
    }
}

#if FFTLIB == FFTW

fftwf_plan fft_plan;

void init_fft() {
    samples = (fc_complex*) fftwf_malloc(sizeof(fc_complex) * block_size);
    fft = (fc_complex*) fftwf_malloc(sizeof(fc_complex) * block_size);

    if (samples == NULL || fft == NULL) {
        fprintf(stderr, "init fft malloc failed\n");
        exit(1);
    }

    // TODO: load wisdom
    // TODO: configure threading
    
    fft_plan = fftwf_plan_dft_1d(
            block_size,
            reinterpret_cast<fftwf_complex *>(samples),
            reinterpret_cast<fftwf_complex *>(fft),
            FFTW_FORWARD,
            FFTW_MEASURE);

    if (fft_plan == NULL) {
        fprintf(stderr, "failed to create fft plan\n");
        exit(1);
    }

    // TODO: save wisdom
}

void free_fft() {
    fftwf_destroy_plan(fft_plan);
    fftwf_free(samples);
    fftwf_free(fft);
}

void perform_fft() {
    fftwf_execute(fft_plan);
}

#else

void init_fft() {
    samples = (fc_complex*) malloc(sizeof(fc_complex) * block_size);
}

void free_fft() {
    free(samples);
}

#endif

int main() {
    // FILE* in = stdin;
    FILE* in = fopen("test.dat", "rb");
    if (in == NULL) {
        perror("Failed to open input file");
        exit(1);
    }

    init_buffers();

    while (read_next_block(in)) {
        convert_raw_to_complex();
        perform_fft();
    }

    free_fft();

    if (in != stdin) {
        fclose(in);
    }
}
