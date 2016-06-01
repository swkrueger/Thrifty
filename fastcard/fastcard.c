/**
 * FastCarD: Fast Carrier Detection
 * 
 * Features:
 *  - fast IO and raw-to-complex conversion
 *  - fast fft
 *    + fftw if USE_FFTW (default)
 *    + gpufft if USE_GPUFFT (rpi)
 *  - volk for abs
 *  - fast md5 (with libb64)
 *
 *  Dependencies:
 *   - fftw / gpufft
 *   - libvolk
 **/

#include <endian.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <sys/time.h>
#include <time.h>

#include <error.h>
#include <argp.h>

#include "configuration.h"
#include "lib/base64.h"

#ifdef USE_FFTW
#include <fftw3.h>
#endif

#ifdef USE_GPUFFT
#include <unistd.h>
#include "lib/gpu_fft/gpu_fft.h"
#include "lib/gpu_fft/mailbox.h"
#endif

#ifdef USE_VOLK
#include <volk/volk.h>
#endif

#ifndef __STDC_IEC_559_COMPLEX__
#error Complex numbers not supported
#endif

typedef float complex fc_complex;

// Settings
#define block_size_log2 13
int block_size = 1<<block_size_log2; // 8196
int history_size = 2085;

float threshold_constant = 12;
float threshold_snr = 0;
int carrier_freq_min = 7897;  // -80 kHz
int carrier_freq_max = 7917;  // -75 kHz

char *input_file = "";
char *output_file = NULL;
char *wisdom_file = "fastcard.fftw_wisdom";

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

    base64 = (char*) malloc((2*block_size+2)/3*4 + 1);

    if (raw_samples == NULL || fft_mag == NULL || base64 == NULL) {
        fprintf(stderr, "init buffers failed\n");
        exit(1);
    }

    init_fft();

    // normalize carrier_freq_{min,max}
    if (carrier_freq_min < 0) {
        carrier_freq_min = block_size + carrier_freq_min;
    }
    if (carrier_freq_max < 0) {
        carrier_freq_max = block_size + carrier_freq_max;
    }
    if (carrier_freq_max < carrier_freq_min) {
        int t = carrier_freq_max;
        carrier_freq_max = carrier_freq_min;
        carrier_freq_min = t;
    }
    // TODO: min < 0 && max > 0 will break (e.g. -1-5 -> 5-8193)
    // TODO: check that carrier_freq is in range (< block_size)

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

#ifdef USE_FFTW

fftwf_plan fft_plan;

void init_fft() {
    samples = (fc_complex*) fftwf_malloc(sizeof(fc_complex) * block_size);
    fft = (fc_complex*) fftwf_malloc(sizeof(fc_complex) * block_size);

    if (samples == NULL || fft == NULL) {
        fprintf(stderr, "init fft malloc failed\n");
        exit(1);
    }

    int r = fftwf_import_wisdom_from_filename(wisdom_file);
    if (r == 0) {
        fprintf(stderr, "failed to import wisdom file\n");
    }

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

    r = fftwf_export_wisdom_to_filename(wisdom_file);
    if (r == 0) {
        fprintf(stderr, "failed to export wisdom file\n");
    }
}

void free_fft() {
    fftwf_destroy_plan(fft_plan);
    fftwf_free(samples);
    fftwf_free(fft);
}

void perform_fft() {
    fftwf_execute(fft_plan);
}

#endif
#ifdef USE_GPUFFT

int mbox;
struct GPU_FFT *fft_state;

static char* gpufft_errors[] = {
    "Unable to enable V3D. Please check your firmware is up to date.\n", // -1
    "fft size not supported. Try between 8 and 21.\n",                   // -2
    "Out of memory.  Try a smaller batch or increase GPU memory.\n",     // -3
    "Unable to map Videocore peripherals into ARM memory space.\n"       // -4
};

void init_fft() {
    samples = (fc_complex*) malloc(sizeof(fc_complex) * block_size);
    int mbox = mbox_open();

    int ret = gpu_fft_prepare(mbox, block_size_log2, GPU_FFT_FWD, 1, &fft_state);

    if (ret <= -1 && ret >= -4) {
        fputs(gpufft_errors[-1-ret], stderr);
        exit(1);
    }
}

void free_fft() {
    free(samples);
}

void perform_fft() {
    memcpy(fft_state->in, samples, sizeof(fc_complex) * block_size);
    // usleep(1); // yield to OS
    gpu_fft_execute(fft_state);
    fft = (fc_complex*) fft_state->out;
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
    float sum = 0; // todo: volk_malloc
    if (threshold_snr == 0) {
        volk_32fc_magnitude_32f_u(
                fft_mag + carrier_freq_min,
                fft + carrier_freq_min,
                carrier_freq_max - carrier_freq_min + 1);
    } else {
        volk_32fc_magnitude_32f_u(fft_mag, fft, block_size);
        volk_32f_accumulator_s32f(&sum, fft_mag, block_size);
    }

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
    Base64encode(base64, input, block_size * 2);
}

/* Parse arguments */
const char *argp_program_version = "fastcard " VERSION_STRING;
static char doc[] = "FastCarD: Fast Carrier Detection";
static struct argp_option options[] = {
    {"input",  'i', "<FILE>", 0,
        "Input file (blank or omit for stdin)", 0},
    {"output", 'o', "<FILE>", 0,
        "Output detections to file (blank or '-' for stdout)", 0},
    {"carrier", 'c', "<min>-<max>", 0,
        "Window of frequency bins used for carrier detection.", 1},
    {"threshold", 't', "<constant>c<snr>s", 0,
        "Carrier detection theshold.", 1},
    {0, 0, 0, 0, 0, 0}
};

static bool parse_carrier_str(char *arg) {
    int r = sscanf(arg, "%d-%d", &carrier_freq_min, &carrier_freq_max);

    if (r == 1) {
        carrier_freq_max = carrier_freq_min;
    } else if (r != 2) {
        fprintf(stderr, "Argument '--carrier' contains an invalid value.\n");
        return false;
    }
    
    return true;
}

static bool parse_theshold_str(char *arg) {
    float f;
    int n;
    
    bool got_constant = false;
    bool got_snr = false;

    threshold_constant = 0;
    threshold_snr = 0;

    while (sscanf(arg, "%f%n", &f, &n) == 1) {
        arg += n;
        switch (*arg) {
            case 'c':
                arg += 1;
            case '\0':
                if (got_constant) {
                    fprintf(stderr, "Argument '--threshold' contains more than "
                                    "one value for constant.\n");
                    return false;
                }
                threshold_constant = f;
                got_constant = true;
                break;
            case 's':
                if (got_snr) {
                    fprintf(stderr, "Argument '--threshold' contains more than "
                                    "one value for SNR.\n");
                    return false;
                }
                threshold_snr = f;
                arg +=1;
                got_snr = true;
                break;
        }
    }

    if (*arg != '\0') {
        fprintf(stderr, "Argument '--threshold' contains an invalid value.\n");
        return false;
    }

    return true;
}

/* Parse a single option. */
static error_t parse_opt (int key, char *arg, struct argp_state *state) {
  switch (key) {
    case 'i': input_file = arg; break;
    case 'o': output_file = arg; break;
    case 'c':
        if (!parse_carrier_str(arg)) argp_usage(state);
        break;
    case 't':
        if (!parse_theshold_str(arg)) argp_usage(state);
        break;
    // We don't take any arguments
    case ARGP_KEY_ARG: argp_usage(state); break;
    default:
        return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

static struct argp argp = {options, parse_opt, NULL, doc, NULL, NULL, NULL};

void parse_args(int argc, char **argv) {
    argp_parse(&argp, argc, argv, 0, 0, 0);
}

int main(int argc, char **argv) {
    parse_args(argc, argv);

    FILE* in;
    if (strlen(input_file) == 0 || strcmp(input_file, "-") == 0) {
        in = stdin;
    } else {
        in = fopen(input_file, "rb");
        if (in == NULL) {
            perror("Failed to open input file");
            exit(1);
        }
    }

    FILE* out = NULL;
    if (output_file != NULL) {
        if (strlen(output_file) == 0 || strcmp(output_file, "-") == 0) {
            out = stdout;
        } else {
            out = fopen(output_file, "w");
            if (out == NULL) {
                perror("Failed to open output file");
                exit(1);
            }
        }
    }

    init_buffers();

    fprintf(stderr, "carrier bin window: min = %d; max = %d\n",
            carrier_freq_min, carrier_freq_max);
    fprintf(stderr, "threshold: constant = %g; snr = %g\n",
            threshold_constant, threshold_snr);

    if (out != NULL) {
        fprintf(out,
                "# arguments: { carrier_bin: '%d-%d', threshold: '%gc+%gs', "
                "block_size: %d, history_size: %d }\n",
                carrier_freq_min, carrier_freq_max,
                threshold_constant, threshold_snr,
                block_size, history_size);
        fprintf(out, "# tool: 'fastcard " VERSION_STRING "'\n");

        struct timeval tv;
        gettimeofday(&tv, NULL);
        fprintf(out, "# start_time: %ld.%06ld\n", tv.tv_sec, tv.tv_usec);
    }

    carrier_detection_t d;
	struct timespec ts;

    int i = 0;
    while (read_next_block(in)) {
        convert_raw_to_complex();
        perform_fft();
        if (detect_carrier(&d)) {
            // Get timestamp
            // This might impact performance negatively
            // (https://stackoverflow.com/questions/6498972/)
            clock_gettime(CLOCK_MONOTONIC, &ts);

            fprintf(stderr,
                    "block #%d: mag[%d] = %.1f (thresh = %.1f)\n",
                    i, d.argmax, d.max, d.threshold);

            if (out != NULL) {
                base64_encode();
                fprintf(out, "%ld.%09ld %d %s\n", ts.tv_sec, ts.tv_nsec, i, base64);
            }
        }
        ++i;
    }

    free_buffers();

    if (in != stdin) {
        fclose(in);
    }

    if (out != NULL && out != stdout) {
        fclose(out);
    }
}
