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
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

#include <error.h>
#include <argp.h>

#include "configuration.h"
#include "lib/base64.h"

#ifdef USE_FFTW
#include <fftw3.h>
#endif /* USE_FFTW */

#ifdef USE_GPUFFT
#include <unistd.h>
#include "lib/gpu_fft/gpu_fft.h"
#include "lib/gpu_fft/mailbox.h"
#endif /* USE_GPUFFT */

#ifdef USE_VOLK
#include <volk/volk.h>
#endif /* USE_VOLK */

#ifdef USE_LIBRTLSDR
#include <pthread.h>
#include <rtl-sdr.h>
#include "circbuf.h"

#define DEFAULT_FREQUENCY 433830000     // 433.83 MHz
#define DEFAULT_SAMPLE_RATE 2400000     // 2.4 Msps
#define DEFAULT_GAIN 0                  // 0 dB

#define CIRCBUF_SIZE (16 * 16384 * 32)  // 8 MiB
#define RTLSDR_BUF_LENGTH (16 * 16384)  // 256 KiB
#define RTLSDR_BUF_NUM (16)             // \_> * 16 = 4 MiB
#endif /* USE_LIBRTLSDR */

#ifndef __STDC_IEC_559_COMPLEX__
#error Complex numbers not supported
#endif

typedef float complex fc_complex;

// Settings
size_t block_size = 16384;  // 2^14
size_t history_size = 5210;

float threshold_constant = 100;
float threshold_snr = 2;
int carrier_freq_min = 0;
int carrier_freq_max = -1;
unsigned blocks_skip = 1;

char *input_file = "";
char *output_file = NULL;

FILE *info = NULL;
bool silent = false;

#ifdef USE_FFTW
char *wisdom_file = "fastcard.fftw_wisdom";
#endif

// Buffers
uint16_t *raw_samples;
fc_complex *samples;
fc_complex *fft;
float *fft_power;
fc_complex lut[0x10000];
char *base64;

bool volatile keep_running = true;

#ifdef USE_LIBRTLSDR
rtlsdr_dev_t *sdr_dev = NULL;
circbuf_t *circbuf = NULL;
pthread_t sdr_thread;
int sdr_return_code;

// WARNING: keep_running is accessed by multiple threads without a mutex.
// FIXME: potential race condition
bool volatile sdr_running = false;

uint32_t sdr_frequency = DEFAULT_FREQUENCY;
uint32_t sdr_sample_rate = DEFAULT_SAMPLE_RATE;
int sdr_gain = DEFAULT_GAIN;
uint32_t sdr_dev_index = 0;
#endif

void signal_handler(int signo) {
    (void)signo;  // unused
    keep_running = false;

#ifdef USE_LIBRTLSDR
    if (circbuf != NULL) {
        circbuf_cancel(circbuf);
    }
#endif
}

void info_out(const char * format, ...) {
    if (!silent && info != NULL) {
        va_list args;
        va_start(args, format);
        vfprintf(info, format, args);
        va_end(args);
    }
}

#ifdef USE_LIBRTLSDR

// Copied from rtlsdr/src/convenience/convenience.c
// <copy>
int nearest_gain(rtlsdr_dev_t *dev, int target_gain) {
    int i, err1, err2, count, nearest;
    int* gains;
    count = rtlsdr_get_tuner_gains(dev, NULL);
    if (count <= 0) {
        return 0;
    }
    gains = malloc(sizeof(int) * count);
    count = rtlsdr_get_tuner_gains(dev, gains);
    nearest = gains[0];
    for (i=0; i<count; i++) {
        err1 = abs(target_gain - nearest);
        err2 = abs(target_gain - gains[i]);
        if (err2 < err1) {
            nearest = gains[i];
        }
    }
    free(gains);
    return nearest;
}

/* standard suffixes */
double atofs(char *s) {
	char last;
	int len;
	double suff = 1.0;
	len = strlen(s);
	last = s[len-1];
	s[len-1] = '\0';
	switch (last) {
		case 'g':
		case 'G':
			suff *= 1e3;
		case 'm':
		case 'M':
			suff *= 1e3;
		case 'k':
		case 'K':
			suff *= 1e3;
			suff *= atof(s);
			s[len-1] = last;
			return suff;
	}
	s[len-1] = last;
	return atof(s);
}
// </copy>

static void sdr_callback(unsigned char *buf, uint32_t len, void *ctx) {
    if (ctx) {
        if (!sdr_running) return;
        circbuf_put((circbuf_t*) ctx, (char*) buf, len);
    }
}

void sdr_free();

bool sdr_init() {
    circbuf = NULL;
    sdr_dev = NULL;

    uint32_t device_count = rtlsdr_get_device_count();
    if (device_count == 0) {
        fprintf(stderr, "No supported RTL-SDR devices found.\n");
        goto fail;
    }
    if (sdr_dev_index >= device_count) {
        fprintf(stderr, "RTL-SDR #%d not found\n", device_count);
        goto fail;
    }

    int r = rtlsdr_open(&sdr_dev, (uint32_t)sdr_dev_index);
    if (r < 0) {
		fprintf(stderr, "Failed to open RTL-SDR device #%d.\n",
                sdr_dev_index);
        sdr_dev = NULL;
        goto fail;
    }

    // set sample rate
    r = rtlsdr_set_sample_rate(sdr_dev, sdr_sample_rate);
    if (r < 0) {
		fprintf(stderr, "Failed to set sample rate.\n");
        goto fail;
    }

    // set center frequency
	r = rtlsdr_set_center_freq(sdr_dev, sdr_frequency);
    if (r < 0) {
		fprintf(stderr, "Failed to set center frequency.\n");
        goto fail;
    }

    // manual gain mode
    r = rtlsdr_set_tuner_gain_mode(sdr_dev, 1);
    if (r < 0) {
        fprintf(stderr, "Failed to enable manual gain.\n");
        goto fail;
    }

    // set gain
    sdr_gain = nearest_gain(sdr_dev, sdr_gain);
    r = rtlsdr_set_tuner_gain(sdr_dev, sdr_gain);
	if (r != 0) {
		fprintf(stderr, "Failed to set tuner gain.\n");
        goto fail;
    }

    // reset_buffer
	r = rtlsdr_reset_buffer(sdr_dev);
	if (r < 0) {
		fprintf(stderr, "WARNING: Failed to reset buffers.\n");
    }

    circbuf = circbuf_new(CIRCBUF_SIZE);
    if (circbuf == NULL) {
        fprintf(stderr, "Failed to create circular buffer\n");
        return false;
    }

    return true;

fail:
    sdr_free();
    return false;
}

void *sdr_routine(void * args) {
    (void)args; // ignore unused argument

    int r = rtlsdr_read_async(
            sdr_dev, sdr_callback, (void *)circbuf,
            RTLSDR_BUF_NUM, RTLSDR_BUF_LENGTH);

    if (sdr_running) {
        // Premature exit -- an error occurred
        circbuf_cancel(circbuf);
    } else {
        r = 0;
    }

    sdr_return_code = r;
    return NULL;
}

void sdr_free() {
    if (sdr_dev != NULL) {
        rtlsdr_close(sdr_dev);
    }
    if (circbuf != NULL) {
        circbuf_free(circbuf);
    }
}

bool sdr_start() {
    // Create RTL-SDR thread
    int r = pthread_create(&sdr_thread, NULL, sdr_routine, NULL);
    if (r != 0) {
        return false;
    }

    sdr_running = true;
    return true;
}

int sdr_stop() {
    sdr_running = false;
    circbuf_cancel(circbuf); // deadlock
    rtlsdr_cancel_async(sdr_dev);

    // wait for thread to finish
    pthread_join(sdr_thread, NULL);
    return sdr_return_code;
}

bool rtl_read(uint16_t *dst, size_t num_samples) {
    return circbuf_get(circbuf, (char *)dst, 2 * num_samples);
}
#endif

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

// normalize carrier_freq_{min,max}
void normalize_carrier_freq() {
    if (carrier_freq_min < 0 && carrier_freq_max >= 0) {
        fprintf(stderr, "Carrier frequency window range not supported.\n");
        exit(1);
    }
    if (carrier_freq_min < 0) {
        carrier_freq_min = block_size + carrier_freq_min;
    }
    if (carrier_freq_max < 0) {
        carrier_freq_max = block_size + carrier_freq_max;
    }
    if ((size_t)carrier_freq_min >= block_size
            || (size_t)carrier_freq_max >= block_size) {
        fprintf(stderr, "Carrier frequency window out of range.\n");
        exit(1);
    }
    if (carrier_freq_max < carrier_freq_min) {
        int t = carrier_freq_max;
        carrier_freq_max = carrier_freq_min;
        carrier_freq_min = t;
    }
    // TODO: min < 0 && max > 0 will break (e.g. -1-5 -> 5-8193)
    // TODO: check that carrier_freq is in range (< block_size)

}

void init_buffers() {
    raw_samples = (uint16_t*) malloc(block_size * sizeof(uint16_t));
    for (size_t i = 0; i < block_size; ++i) raw_samples[i] = 127;

#ifdef USE_VOLK
    size_t alignment = volk_get_alignment();
    fft_power = (float*) volk_malloc(block_size * sizeof(float), alignment);
#else
    fft_power = (float*) malloc(block_size * sizeof(float), alignment);
#endif

    base64 = (char*) malloc((2*block_size+2)/3*4 + 1);

    if (raw_samples == NULL || fft_power == NULL || base64 == NULL) {
        fprintf(stderr, "init buffers failed\n");
        exit(1);
    }

    init_fft();

    generate_lut();
}

void free_buffers() {
    free_fft();
    free(raw_samples);
    volk_free(fft_power);
    free(base64);
}

bool read_next_block(FILE *f) {
    // copy history
    size_t b = block_size - history_size;
    memcpy(raw_samples,
           raw_samples + b,
           history_size * 2);

#ifdef USE_LIBRTLSDR
    if (f == NULL) { // read from RTL-SDR
        return rtl_read(raw_samples + history_size, b);
    }
#endif /* USE_LIBRTLSDR */

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
    for (size_t i = 0; i < block_size; ++i) {
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

    if (!silent) {
        info_out("FFTW plan:\n");
        fftwf_fprint_plan(fft_plan, info);
        info_out("\n\n");
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
    float noise;
} carrier_detection_t;

bool detect_carrier(carrier_detection_t *d) {
    // calculate magnitude
#ifdef USE_VOLK
    float sum = 0; // todo: volk_malloc
    if (threshold_snr == 0) {
        volk_32fc_magnitude_squared_32f_u(
                fft_power + carrier_freq_min,
                fft + carrier_freq_min,
                carrier_freq_max - carrier_freq_min + 1);
    } else {
        volk_32fc_magnitude_squared_32f_a(fft_power, fft, block_size);
        volk_32f_accumulator_s32f(&sum, fft_power, block_size);
    }

    // volk_32fc_magnitude_squared_32f is faster than volk_32fc_magnitude_32f

    uint16_t argmax; // todo: volk_malloc
    volk_32f_index_max_16u(
            &argmax,
            fft_power + carrier_freq_min,
            carrier_freq_max - carrier_freq_min + 1);
    argmax += carrier_freq_min;
    float max = fft_power[argmax];

#else
    for (int i = 0; i < block_size; ++i) {
        fft_power[i] = fft[i] * conj(fft[i]);
    }

    float sum = 0;
    for (int i = 0; i < block_size; ++i) {
        sum += fft_power[i];
    }

    float max = 0;
    int argmax;
    for (int i = carrier_freq_min; i <= carrier_freq_max; ++i) {
        if (fft_power[i] > max) {
            argmax = i;
            max = fft_power[i];
        }
    }
#endif

    float noise_power = (sum == 0) ? 0 : (sum - max) / (block_size - 1);
    float threshold = threshold_constant + threshold_snr * noise_power;

    if (max > threshold) {
        if (d != NULL) {
            d->argmax = argmax;
            d->max = max;
            d->threshold = threshold;
            d->noise = noise_power;
        }
        return true;
    }

    return false;
}

void base64_encode() {
    const char* input = (const char*) raw_samples;
    Base64encode(base64, input, block_size * 2);
}

void process(FILE* in, FILE* out) {
    carrier_detection_t d;
    struct timeval ts;

#ifdef USE_LIBRTLSDR
    if (in == NULL) {
        if (!sdr_start()) {
            fprintf(stderr, "Failed to start RTL-SDR\n");
            // TODO: return non-zero exit code
            return;
        }
    }
#endif /* USE_LIBRTLSDR */

    unsigned long i = 0;
    while (read_next_block(in) && keep_running) {
        if (blocks_skip != 0) {
            blocks_skip--;
            continue;
        }

        convert_raw_to_complex();
        perform_fft();
        if (detect_carrier(&d)) {
            // Get coarse timestamp
            // This might impact performance negatively
            // (https://stackoverflow.com/questions/6498972/)
            //
            // TODO: add timestamp when we receive the data, not when it is
            //       being processed.
            gettimeofday(&ts, NULL);

            info_out("block #%lu: mag[%d] = %.1f (thresh = %.1f, noise = %.1f)\n",
                     i, d.argmax, sqrt(d.max),
                     sqrt(d.threshold), sqrt(d.noise));

            if (out != NULL) {
                base64_encode();
                fprintf(out, "%ld.%06ld %lu %s\n",
                        ts.tv_sec, ts.tv_usec, i, base64);
            }
        }
        ++i;
    }

#ifdef USE_LIBRTLSDR
    if (in == NULL) {
        int r = sdr_stop();
        if (r != 0) {
            fprintf(stderr, "\nRTL-SDR library error %d, exiting...\n", r);
            // TODO: non-zero exit code
        }
    }
#endif /* USE_LIBRTLSDR */

    info_out("\nRead %lu blocks.\n", i);
}

/* Parse arguments */
const char *argp_program_version = "fastcard " VERSION_STRING;
static char doc[] = "FastCarD: Fast Carrier Detection\n\n"
    "Takes a stream of raw 8-bit IQ samples from a RTL-SDR, splits it into "
    "fixed-sized blocks, and, if a carrier is detected in a block, outputs the "
    "block ID, timestamp and the block's raw samples encoded in base64.";

static struct argp_option options[] = {
    // I/O
    {0, 0, 0, 0, "I/O settings:", 1},
    {"input",  'i', "<FILE>", 0,
        "Input file with samples "
#ifdef USE_LIBRTLSDR
        "('-' for stdin, 'rtlsdr' for librtlsdr)\n[default: stdin]",
#else /* USE_LIBRTLSDR */
        "('-' for stdin)\n[default: stdin]",
#endif /* USE_LIBRTLSDR */
        1},
    {"output", 'o', "<FILE>", 0,
        "Output card file ('-' for stdout)\n[default: no output]", 1},
#ifdef USE_FFTW
    {"wisdom-file", 'm', "<size>", 0,
        "Wisfom file to use for FFT calculation"
        "\n[default: fastcard.fftw_wisdom]", 1},
#endif

    // Blocks
    {0, 0, 0, 0, "Block settings:", 2},
    {"block-size", 'b', "<size>", 0,
        "Length of fixed-sized blocks, which should be a power of two "
        "[default: 16384]", 2},
    {"history", 'h', "<size>", 0,
        "The number of samples at the beginning of a block that should be "
        "copied from the end of the previous block [default: 5210]", 2},
    {"skip", 'k', "<num_blocks>", 0,
        "Number of blocks to skip while waiting for the SDR to stabilize "
        "[default: 1]", 2},

    // Carrier detection
    {0, 0, 0, 0, "Carrier detection settings:", 3},
    {"carrier-window", 'w', "<min>-<max>", 0,
        "Window of frequency bins used for carrier detection "
        "[default: no window]", 3},
    {"threshold", 't', "<constant>c<snr>s", 0,
        "Carrier detection theshold [default: 100c2s]", 3},

#ifdef USE_LIBRTLSDR
    // Tuner
    {0, 0, 0, 0, "Tuner settings (if input is 'rtlsdr'):", 4},
    {"frequency", 'f', "<hz>", 0,
        "Frequency to tune to [default: 433.83M]", 4},
    {"sample-rate", 's', "<sps>", 0,
        "Sample rate [default: 2.4M]", 4},
    {"gain", 'g', "<db>", 0,
        "Gain [default: 0]", 4},
    {"device-index", 'd', "<index>", 0,
        "RTL-SDR device index [default: 0]", 4},
#endif /* USE_LIBRTLSDR */

    // Misc
    {0, 0, 0, 0, "Miscellaneous:", -1},
    {"quiet", 'q', 0, 0, "Shhh", -1},
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
    char* endptr;

    switch (key) {
        case 'i': input_file = arg; break;
        case 'o': output_file = arg; break;
        case 'm': wisdom_file = arg; break;
        case 'w':
            if (!parse_carrier_str(arg)) argp_usage(state);
            break;
        case 't':
            if (!parse_theshold_str(arg)) argp_usage(state);
            break;
        case 'b':
            block_size = strtoul(arg, &endptr, 10);
            if (endptr == NULL || block_size < 1) argp_usage(state);
            break;
        case 'h':
            history_size = strtoul(arg, &endptr, 10);
            if (endptr == NULL || history_size < 1) argp_usage(state);
            break;
        case 'k':
            blocks_skip = strtoul(arg, &endptr, 10);
            if (endptr == NULL || history_size < 1) argp_usage(state);
            break;
        case 'q':
            silent = true;
            break;
#ifdef USE_LIBRTLSDR
        case 'f':
            sdr_frequency = (uint32_t)atofs(arg);
            break;
        case 'g':
            sdr_gain = (int)(atof(arg) * 10); // unit: tenths of a dB
            break;
        case 's':
            sdr_sample_rate = (uint32_t)atofs(arg);
            break;
        case 'd':
            sdr_dev_index = strtoul(arg, &endptr, 10);
            if (endptr == NULL || block_size < 1) argp_usage(state);
            break;
#endif /* USE_LIBRTLSDR */
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

    if (history_size > block_size) {
        fprintf(stderr, "History length cannot be larger than block size.\n");
        exit(1);
    }

    FILE* in;
    if (strlen(input_file) == 0 || strcmp(input_file, "-") == 0) {
        in = stdin;
#ifdef USE_LIBRTLSDR
    } else if (strcmp(input_file, "rtlsdr") == 0) {
        in = NULL;
#endif /* USE_LIBRTLSDR */
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

    if (out == stdout) {
        info = stderr;
    } else {
        info = stdout;
    }

    normalize_carrier_freq();
    init_buffers();

#ifdef USE_LIBRTLSDR
    if (in == NULL) {
        if (!sdr_init()) {
            fprintf(stderr, "Failed to initialize RTL-SDR.\n");
            exit(1);
        }
    }
#endif /* USE_LIBRTLSDR */

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGQUIT, signal_handler);
    signal(SIGPIPE, signal_handler);

    info_out("block size: %zu; history length: %zu\n",
             block_size, history_size);
    info_out("carrier bin window: min = %d; max = %d\n",
             carrier_freq_min, carrier_freq_max);
    info_out("threshold: constant = %g; snr = %g\n\n",
             threshold_constant, threshold_snr);

#ifdef USE_LIBRTLSDR
    if (in == NULL) {
        info_out("tuner:\n"
                 "  center freq = %.06f MHz\n"
                 "  sample rate = %.06f Msps\n"
                 "  gain = %.02f dB\n",
                 sdr_frequency / 1e6, sdr_sample_rate / 1e6, sdr_gain/10.0);
    }
#endif /* USE_LIBRTLSDR */
    fflush(info);

    if (out != NULL) {
        fprintf(out,
                "# arguments: { carrier_bin: '%d-%d', threshold: '%gc+%gs', "
                "block_size: %zu, history_size: %zu }\n",
                carrier_freq_min, carrier_freq_max,
                threshold_constant, threshold_snr,
                block_size, history_size);
#ifdef USE_LIBRTLSDR
        if (in == NULL) {
            fprintf(out, "# tuner: { freq: %u; sample_rate: %u; gain: %02f }\n",
                    sdr_frequency, sdr_sample_rate, sdr_gain/10.0);
        }
#endif /* USE_LIBRTLSDR */
        fprintf(out, "# tool: 'fastcard " VERSION_STRING "'\n");

        struct timeval tv;
        gettimeofday(&tv, NULL);
        fprintf(out, "# start_time: %ld.%06ld\n", tv.tv_sec, tv.tv_usec);
        fflush(out);
    }

    process(in, out);

    if (info != NULL) {
        fflush(info);
    }

    if (out != NULL) {
        fflush(out);
    }

#ifdef USE_LIBRTLSDR
    if (in == NULL) {
        sdr_free();
    }
#endif /* USE_LIBRTLSDR */

    free_buffers();

    if (in != NULL && in != stdin) {
        fclose(in);
    }

    if (out != NULL && out != stdout) {
        fclose(out);
    }

    // TODO: return non-zero exit code when error occurred
}
