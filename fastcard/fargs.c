#include <sys/time.h>

#include "parse.h"
#include "fargs.h"

#define DEFAULT_BLOCK_LEN           16384
#define DEFAULT_HISTORY_LEN         4920
#define DEFAULT_THRESHOLD_CONST     100
#define DEFAULT_THRESHOLD_SNR       2
#define DEFAULT_CARRIER_FREQ_MIN    0
#define DEFAULT_CARRIER_FREQ_MAX    -1
#define DEFAULT_SKIP                1
#define DEFAULT_INPUT_FILE          "-"
#define DEFAULT_WISDOM_FILE         NULL

#define DEFAULT_SDR_FREQ            433830000
#define DEFAULT_SDR_SAMPLE_RATE     2400000
#define DEFAULT_SDR_GAIN            0
#define DEFAULT_SDR_INDEX           0

static char* default_input_file = DEFAULT_INPUT_FILE;
static char* default_wisdom_file = DEFAULT_WISDOM_FILE;


#define ARGP_KEY_CARD 0x01

// note: remember to update FARGS_NUM_OPTIONS
// this argp stuff is a mess
const fargs_option_t fargs_options[] = {
    // I/O
    {0, 0, 0, 0, "I/O settings:", 1},
    {"input",  'i', "<FILE>", 0,
        "Input file with samples "
        "\n('-' for stdin, 'rtlsdr' for librtlsdr)\n[default: stdin]",
        1},
    {"card", ARGP_KEY_CARD, 0, 0,
        "Input is a .card file instead of binary data", 1},
    {"wisdom-file", 'm', "<FILE>", 0,
        "Wisfom file to use for FFT calculation"
        "\n[default: don't use wisdom file]", 1},

    // Blocks
    {0, 0, 0, 0, "Block settings:", 2},
    {"block-len", 'b', "<length>", 0,
        "Length of fixed-sized blocks, which should be a power of two "
        "[default: 16384]", 2},
    {"history", 'h', "<length>", 0,
        "The number of samples at the beginning of a block that should be "
        "copied from the end of the previous block [default: 4920]", 2},
    {"skip", 'k', "<num_blocks>", 0,
        "Number of blocks to skip while waiting for the SDR to stabilize "
        "[default: 1]", 2},

    // Tuner
    {0, 0, 0, 0, "Tuner settings (if input is 'rtlsdr'):", 3},
    {"frequency", 'f', "<hz>", 0,
        "Frequency to tune to [default: 433.83M]", 3},
    {"sample-rate", 's', "<sps>", 0,
        "Sample rate [default: 2.4M]", 3},
    {"gain", 'g', "<db>", 0,
        "Gain [default: 0]", 3},
    {"device-index", 'd', "<index>", 0,
        "RTL-SDR device index [default: 0]", 3},

    // Carrier detection
    {0, 0, 0, 0, "Carrier detection settings:", 4},
    {"carrier-window", 'w', "<min>-<max>", 0,
        "Window of frequency bins used for carrier detection "
        "[default: no window (0--1)]", 4},
    {"threshold", 't', "<constant>c<snr>s", 0,
        "Carrier detection theshold [default: 100c2s]", 4},

    // Misc
    {0, 0, 0, 0, "Miscellaneous:", -1},
    {"quiet", 'q', 0, 0, "Shhh", -1},
    {0, 0, 0, 0, 0, 0}
};

fargs_t* fargs_new() {
    fargs_t* fargs = malloc(sizeof(fargs_t));
    if (fargs == NULL) {
        return NULL;
    }
    
    fargs->block_len = DEFAULT_BLOCK_LEN;
    fargs->history_len = DEFAULT_HISTORY_LEN;
    
    fargs->threshold_const = DEFAULT_THRESHOLD_CONST;
    fargs->threshold_snr = DEFAULT_THRESHOLD_SNR;
    fargs->carrier_freq_min = DEFAULT_CARRIER_FREQ_MIN;
    fargs->carrier_freq_max = DEFAULT_CARRIER_FREQ_MAX;;
    fargs->skip = DEFAULT_SKIP;
    
    fargs->input_file = default_input_file;
    fargs->wisdom_file = default_wisdom_file;
    fargs->input_card = false;

    fargs->sdr_freq = DEFAULT_SDR_FREQ;
    fargs->sdr_sample_rate = DEFAULT_SDR_SAMPLE_RATE;
    fargs->sdr_gain = DEFAULT_SDR_GAIN;
    fargs->sdr_dev_index = DEFAULT_SDR_INDEX;

    fargs->silent = false;

    return fargs;
}

int fargs_parse_opt(fargs_t *fargs,
                    int key,
                    char *arg) {
    char* endptr;

    switch (key) {
        case ARGP_KEY_CARD:
            fargs->input_card = true;
        case 'i': fargs->input_file = arg; break;
        case 'm': fargs->wisdom_file = arg; break;
        case 'w':
            if (!parse_carrier_str(arg,
                                   &fargs->carrier_freq_min,
                                   &fargs->carrier_freq_max)) {
                return FARGS_INVALID_VALUE;
            }
            break;
        case 't':
            if (!parse_theshold_str(arg,
                                    &fargs->threshold_const,
                                    &fargs->threshold_snr)) {
                return FARGS_INVALID_VALUE;
            }
            break;
        case 'b':
            fargs->block_len = strtoul(arg, &endptr, 10);
            if (endptr == NULL || fargs->block_len < 1) {
                return FARGS_INVALID_VALUE;
            }
            break;
        case 'h':
            fargs->history_len = strtoul(arg, &endptr, 10);
            if (endptr == NULL || fargs->history_len < 1) {
                return FARGS_INVALID_VALUE;
            }
            break;
        case 'k':
            fargs->skip = strtoul(arg, &endptr, 10);
            if (endptr == NULL || fargs->history_len < 1) {
                return FARGS_INVALID_VALUE;
            }
            break;
        case 'q':
            fargs->silent = true;
            break;
        case 'f':
            fargs->sdr_freq = (uint32_t)parse_si_float(arg);
            break;
        case 'g':
            fargs->sdr_gain = (int)(atof(arg) * 10); // unit: tenths of a dB
            break;
        case 's':
            fargs->sdr_sample_rate = (uint32_t)parse_si_float(arg);
            break;
        case 'd':
            fargs->sdr_dev_index = strtoul(arg, &endptr, 10);
            if (endptr == NULL) {
                return FARGS_INVALID_VALUE;
            }
            break;
        default:
            return FARGS_UNKNOWN;
    }
    return 0;
}

void fargs_print_summary(fargs_t *fa, FILE* out, bool sdr) {
    fprintf(out, "block size: %zu; history length: %zu\n",
            fa->block_len, fa->history_len);
    fprintf(out, "carrier bin window: min = %d; max = %d\n",
            fa->carrier_freq_min, fa->carrier_freq_max);
    fprintf(out, "threshold: constant = %g; snr = %g\n\n",
            fa->threshold_const, fa->threshold_snr);

    if (sdr) {
        fprintf(out,
                "tuner:\n"
                "  center freq = %.06f MHz\n"
                "  sample rate = %.06f Msps\n"
                "  gain = %.02f dB\n\n",
                fa->sdr_freq / 1e6,
                fa->sdr_sample_rate / 1e6,
                fa->sdr_gain/10.0);
    }
}

void fargs_print_card_header(fargs_t *fa,
                             FILE* out,
                             bool sdr,
                             const char* tool) {
    fprintf(out,
            "# arguments: { carrier_bin: '%d-%d', threshold: '%gc+%gs', "
            "block_size: %zu, history_size: %zu }\n",
            fa->carrier_freq_min, fa->carrier_freq_max,
            fa->threshold_const, fa->threshold_snr,
            fa->block_len, fa->history_len);
    if (sdr) {
        fprintf(out, "# tuner: { freq: %u; sample_rate: %u; gain: %02f }\n",
                fa->sdr_freq, fa->sdr_sample_rate, fa->sdr_gain/10.0);
    }
    fprintf(out, "# tool: '%s'\n", tool);

    struct timeval tv;
    gettimeofday(&tv, NULL);
    fprintf(out, "# start_time: %ld.%06ld\n", tv.tv_sec, tv.tv_usec);
    fflush(out);
}
