// FastCarD: Fast Carrier Detection

#include <endian.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <b64/cencode.h>

#define USE_VOLK

#define FFTLIB FFTW

#if FFTLIB == FFTW
#include <fftw3.h>
#else
#error "Unknown FFT library"
#endif

#ifdef USE_VOLK
#include <volk/volk.h>
#endif

#ifndef __STDC_IEC_559_COMPLEX__
#error Complex numbers not supported
#endif

typedef float complex fc_complex;

// Settings
int block_size = 1<<13; // 8196
int history_size = 2085;

float threshold_constant = 5;
float threshold_snr = 3;
int carrier_freq_min = 7897;  // -80 kHz
int carrier_freq_max = 7917;  // -75 kHz

// Buffers
uint16_t *raw_samples;
fc_complex *samples;
fc_complex *fft;
float *fft_mag;
fc_complex lut[0x10000];
char *base64;

void generate_lut() {
    // generate lookup table for raw-to-complex conversion
    for (size_t i = 0; i <= 0xffff; ++i) {
#if __BYTE_ORDER == __LITTLE_ENDIAN
        ((float*)&lut[i])[0] = ((float)(i & 0xff) - 127.4f) * (1.0f/128.0f);
        ((float*)&lut[i])[1] = ((float)(i >> 8) - 127.4f) * (1.0f/128.0f);
#elif __BYTE_ORDER == __BIG_ENDIAN
        ((float*)&lut[i])[0] = ((float)(i >> 8) - 127.4f) * (1.0f/128.0f);
        ((float*)&lut[i])[1] = ((float)(i & 0xff) - 127.4f) * (1.0f/128.0f);
#else
#error "Could not determine endianness"
#endif
    }
}

void init_fft();
void free_fft();

void init_buffers() {
    raw_samples = (uint16_t*) malloc(block_size * sizeof(uint16_t));
    for (int i = 0; i < block_size; ++i) raw_samples[i] = 127;

    // size_t alignment = volk_get_alignment();
    // fft_mag = (fc_complex*) volk_malloc(block_size * sizeof(fc_complex), alignment);
    fft_mag = (float*) malloc(block_size * sizeof(float));

    // real size: (2*block_size+2)/3*4 + 1
    base64 = (char*) malloc((2*block_size+2)/3*4 + block_size);

    if (raw_samples == NULL || fft_mag == NULL || base64 == NULL) {
        fprintf(stderr, "init buffers failed\n");
        exit(1);
    }

    init_fft();

    generate_lut();
}

void free_buffers() {
    free_fft();
    free(raw_samples);
    free(fft_mag);
    free(base64);
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
            (fftwf_complex*) samples,
            (fftwf_complex*) fft,
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

typedef struct {
    unsigned int argmax;
    float max;
    float threshold;
} carrier_detection_t;

bool detect_carrier(carrier_detection_t *d) {
    // calculate magnitude
#ifdef USE_VOLK
    volk_32fc_magnitude_32f_u(fft_mag, fft, block_size);

    float sum; // todo: volk_malloc
    volk_32f_accumulator_s32f(&sum, fft_mag, block_size);

    unsigned int argmax; // todo: volk_malloc
    volk_32f_index_max_16u(
            &argmax,
            fft_mag + carrier_freq_min,
            carrier_freq_max - carrier_freq_min + 1);
    argmax += carrier_freq_min;
    float max = fft_mag[argmax];

#else
    for (int i = 0; i < block_size; ++i) {
        fft_mag[i] = cabsf(fft[i]);
    }

    float sum = 0;
    for (int i = 0; i < block_size; ++i) {
        sum += fft_mag[i];
    }

    float max = 0;
    int argmax;
    for (int i = carrier_freq_min; i <= carrier_freq_max; ++i) {
        if (fft_mag[i] > max) {
            argmax = i;
            max = fft_mag[i];
        }
    }
#endif

    float mean = sum / block_size;
    float threshold = threshold_constant + threshold_snr * mean;

    if (max > threshold) {
        if (d != NULL) {
            d->argmax = argmax;
            d->max = max;
            d->threshold = threshold;
        }
        return true;
    }

    return false;
}

void base64_encode() {
    const char* input = (const char*) raw_samples;
    char* c = base64;
    int cnt = 0;
    base64_encodestate s;

    base64_init_encodestate(&s);
    cnt = base64_encode_block(input, block_size * 2, c, &s);
    c += cnt;
    cnt = base64_encode_blockend(c, &s);
    c += cnt;

    // null-terminate
    *c = 0;
}

int main() {
    // FILE* in = stdin;
    FILE* in = fopen("test.dat", "rb");
    if (in == NULL) {
        perror("Failed to open input file");
        exit(1);
    }

    FILE* out = fopen("out.txt", "w");
    if (out == NULL) {
        perror("Failed to open output file");
        exit(1);
    }

    init_buffers();

    carrier_detection_t d;

    int i = 0;
    while (read_next_block(in)) {
        convert_raw_to_complex();
        perform_fft();
        if (detect_carrier(&d)) {
            fprintf(stderr,
                    "block #%d: mag[%d] = %.1f (thresh = %.1f)\n",
                    i, d.argmax, d.max, d.threshold);

            base64_encode();
            fprintf(out, "%d %s\n", i, base64);
        }
        ++i;
    }

    free_buffers();

    if (in != stdin) {
        fclose(in);
    }

    if (out != stdout) {
        fclose(out);
    }
}
