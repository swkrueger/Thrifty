/**
 * FastCarD: Fast Carrier Detection
 * 
 * Features:
 *  - fast IO and raw-to-complex conversion
 *  - fast fft
 *    + fftw if USE_FFTW (default)
 *    + gpufft if USE_GPUFFT (rpi) -- deprecated
 *  - volk for abs
 *  - fast md5 (with libb64)
 *
 *  Dependencies:
 *   - fftw / gpufft
 *   - libvolk
 **/

#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

#include <argp.h>
#include <volk/volk.h>

#include "configuration.h"
#include "reader.h"
#include "raw_reader.h"
#include "card_reader.h"
#include "rtlsdr_reader.h"
#include "fft.h"
#include "cardet.h"
#include "rawconv.h"
#include "lib/base64.h"
#include "parse.h"

// Settings: default values
static size_t block_size = 16384;  // 2^14
static size_t history_size = 4920;

static float threshold_constant = 100;
static float threshold_snr = 2;
static int carrier_freq_min = 0;
static int carrier_freq_max = -1;
static unsigned blocks_skip = 1;

static char *input_file = "";
static char *output_file = NULL;
static bool input_card = false;

static FILE* in = NULL;
static FILE* out = NULL;   // default: don't output anything
static FILE *info = NULL;
static bool silent = false;

static uint32_t sdr_frequency = 433830000;     // 433.83 MHz
static uint32_t sdr_sample_rate = 2400000;     // 2.4 Msps
static int sdr_gain = 0;  // 0 dB
static uint32_t sdr_dev_index = 0;


/* Parse arguments */
const char *argp_program_version = "fastcard " VERSION_STRING;
static const char doc[] = "FastCarD: Fast Carrier Detection\n\n"
    "Takes a stream of raw 8-bit IQ samples from a RTL-SDR, splits it into "
    "fixed-sized blocks, and, if a carrier is detected in a block, outputs the "
    "block ID, timestamp and the block's raw samples encoded in base64.";

#define ARGP_KEY_CARD 0x01

static const struct argp_option options[] = {
    // I/O
    {0, 0, 0, 0, "I/O settings:", 1},
    // TODO: --card (input is a card file)
    {"card", ARGP_KEY_CARD, 0, 0,
        "Input is a .card file instead of binary data", 1},
    {"input",  'i', "<FILE>", 0,
        "Input file with samples "
        "('-' for stdin, 'rtlsdr' for librtlsdr)\n[default: stdin]",
        1},
    {"output", 'o', "<FILE>", 0,
        "Output card file ('-' for stdout)\n[default: no output]", 1},

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

    // Misc
    {0, 0, 0, 0, "Miscellaneous:", -1},
    {"quiet", 'q', 0, 0, "Shhh", -1},
    {0, 0, 0, 0, 0, 0}
};

void info_out(const char * format, ...) {
    if (!silent && info != NULL) {
        va_list args;
        va_start(args, format);
        vfprintf(info, format, args);
        va_end(args);
    }
}


/* Parse a single option. */
static error_t parse_opt (int key, char *arg, struct argp_state *state) {
    char* endptr;

    switch (key) {
        case ARGP_KEY_CARD:
            input_card = true;
        case 'i': input_file = arg; break;
        case 'o': output_file = arg; break;
        case 'w':
            if (!parse_carrier_str(arg,
                                   &carrier_freq_min,
                                   &carrier_freq_max)) {
                argp_usage(state);
            }
            break;
        case 't':
            if (!parse_theshold_str(arg,
                                    &threshold_constant,
                                    &threshold_snr)) {
                argp_usage(state);   
            }
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
        case 'f':
            sdr_frequency = (uint32_t)parse_si_float(arg);
            break;
        case 'g':
            sdr_gain = (int)(atof(arg) * 10); // unit: tenths of a dB
            break;
        case 's':
            sdr_sample_rate = (uint32_t)parse_si_float(arg);
            break;
        case 'd':
            sdr_dev_index = strtoul(arg, &endptr, 10);
            if (endptr == NULL || block_size < 1) argp_usage(state);
            break;
        // We don't take any arguments
        case ARGP_KEY_ARG: argp_usage(state); break;
        default:
            return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static struct argp argp = {options, parse_opt, NULL, doc, NULL, NULL, NULL};

int parse_args(int argc, char **argv) {
    argp_parse(&argp, argc, argv, 0, 0, 0);

    if (history_size > block_size) {
        fprintf(stderr, "History length cannot be larger than block size.\n");
        return -1;
    }

    if (strlen(input_file) == 0 || strcmp(input_file, "-") == 0) {
        in = stdin;
    } else if (strcmp(input_file, "rtlsdr") == 0) {
        in = NULL;
    } else {
        in = fopen(input_file, "rb");
        if (in == NULL) {
            perror("Failed to open input file");
            return -1;
        }
    }

    out = NULL;
    if (output_file != NULL) {
        if (strlen(output_file) == 0 || strcmp(output_file, "-") == 0) {
            out = stdout;
        } else {
            out = fopen(output_file, "w");
            if (out == NULL) {
                perror("Failed to open output file");
                return -1;
            }
        }
    }

    if (out == stdout) {
        info = stderr;
    } else {
        info = stdout;
    }

    return 0;
}

void print_header_info() {
    info_out("block size: %zu; history length: %zu\n",
             block_size, history_size);
    info_out("carrier bin window: min = %d; max = %d\n",
             carrier_freq_min, carrier_freq_max);
    info_out("threshold: constant = %g; snr = %g\n\n",
             threshold_constant, threshold_snr);

    if (in == NULL) {
        info_out("tuner:\n"
                 "  center freq = %.06f MHz\n"
                 "  sample rate = %.06f Msps\n"
                 "  gain = %.02f dB\n",
                 sdr_frequency / 1e6, sdr_sample_rate / 1e6, sdr_gain/10.0);
    }

    fflush(info);
}

//////////////////////////////////////////////////////////////////////////////

static block_t* block;
static reader_t* reader;
static complex float* samples;
static complex float* fft;
static float* fft_power;
static fft_state_t* samples_to_fft;
static char *base64;

static rawconv_t rawconv;
static cardet_settings_t cardet_settings;
static bool volatile keep_running = true;

int fastcard_init() {
    // ensure free will only free the variables that have been initialized
    block = NULL;
    reader = NULL;
    samples_to_fft = NULL;
    fft_power = NULL;
    base64 = NULL;

    // init stuff
    rawconv_init(&rawconv);

    block = reader_block_new(block_size);
    if (block == NULL) {
        return -1;
    }

    reader_settings_t reader_settings;
    reader_settings.output = block;
    reader_settings.block_size = block_size;
    reader_settings.history_size = history_size;

    if (in == NULL) {
        rtlsdr_settings_t sdr_settings;
        sdr_settings.sample_rate = sdr_sample_rate;
        sdr_settings.gain = sdr_gain;
        sdr_settings.frequency = sdr_frequency;
        sdr_settings.dev_index = sdr_dev_index;

        reader = rtlsdr_reader_new(reader_settings, &sdr_settings);
        sdr_gain = sdr_settings.gain;  // get exact gain
    } else {
        if (input_card) {
            reader = card_reader_new(reader_settings, in);
            // don't skip blocks when reading from .card file
            blocks_skip = 0;
        } else {
            reader = raw_reader_new(reader_settings, in);
        }
    }
    samples_to_fft = fft_new(block_size, true);
    samples = fft_get_input(samples_to_fft);
    fft = fft_get_output(samples_to_fft);
    size_t alignment = volk_get_alignment();
    fft_power = (float*) volk_malloc(block_size * sizeof(float), alignment);
    if (fft_power == NULL) {
        return -1;
    }

    cardet_settings.threshold_constant = threshold_constant;
    cardet_settings.threshold_snr = threshold_snr;
    cardet_settings.carrier_freq_min = carrier_freq_min;
    cardet_settings.carrier_freq_max = carrier_freq_max;
    cardet_settings.fft_len = block_size;
    if (cardet_normalize_window(&cardet_settings) != 0) {
        return -1;
    }

    base64 = (char*) malloc((2*block_size+2)/3*4 + 1);
    if (base64 == NULL) {
        return -1;
    }

    return 0;
}

void fastcard_free() {
    reader_block_free(block);
    reader_free(reader);
    fft_free(samples_to_fft);
    free(fft_power);
    free(base64);
}

int fastcard_process() {
    int ret = 0;
    cardet_detection_t detection;

    reader_start(reader);

    if (blocks_skip > 0) {
        block->index = ((int64_t)blocks_skip)*-1 - 1;
        info_out("\nSkipping %u block(s)... ", blocks_skip);
        fflush(info);
    }

    keep_running = true;

    unsigned cnt = 0;
    unsigned skip = blocks_skip;

    while (keep_running) {
        ret = reader_next(reader);

        if (ret != 0) {
            if (ret != -1) {
                fprintf(stderr, "reader_next() failed\n");
            }
            break;
        }

        if (skip > 0) {
            --skip;

            if (skip == 0) {
                info_out("done\n\n");
                fflush(info);
            }

            continue;
        }

        rawconv_to_complex(&rawconv, samples, block->raw_samples, block_size);
        fft_perform(samples_to_fft);
        volk_32fc_magnitude_squared_32f_a(fft_power, fft, block_size);
        bool detected = cardet_detect(&cardet_settings,
                                      &detection,
                                      fft_power);
        if (detected) {
            info_out("block #%ld: mag[%d] = %.1f (thresh = %.1f, noise = %.1f)\n",
                     block->index, detection.argmax, sqrt(detection.max),
                     sqrt(detection.threshold), sqrt(detection.noise));

            if (out != NULL) {
                Base64encode(base64,
                             (const char*) block->raw_samples,
                             block_size * 2);
                fprintf(out,
                        "%ld.%06ld %lu %s\n",
                        block->timestamp.tv_sec,
                        block->timestamp.tv_usec,
                        block->index,
                        base64);
            }
        }

        ++cnt;
    }

    // TODO: use separate counter!

    info_out("\nRead %ld blocks.\n", cnt);

    reader_stop(reader);
    if (in == NULL) {
        // print SDR buffer histogram
        rtlsdr_reader_print_histogram(reader, info);
    }

    return ret;
}

void fastcard_cancel() {
    keep_running = false;
    reader_cancel(reader);
}

void signal_handler(int signo) {
    (void)signo;  // unused
    fastcard_cancel();
}


int main(int argc, char **argv) {
    if (parse_args(argc, argv) != 0) {
        exit(1);
    }

    // TODO: set blocks_skip = 0 if input is a card file

    int exit_code = 0;

    exit_code = fastcard_init();
    if (exit_code != 0) {
        goto free;
    }

    print_header_info();

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGQUIT, signal_handler);
    signal(SIGPIPE, signal_handler);

    exit_code = fastcard_process();

free:
    fastcard_free();

    if (info != NULL) {
        fflush(info);
    }

    if (out != NULL) {
        fflush(out);
    }

    if (in != NULL && in != stdin) {
        fclose(in);
    }

    if (out != NULL && out != stdout) {
        fclose(out);
    }

    return exit_code;
}
