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
 *   - librtlsdr
 **/

#include <signal.h>
#include <argp.h>

#include "configuration.h"
#include "lib/base64.h"

#include "fastcard.h"
#include "fargs.h"

const char *argp_program_version = "fastcard " VERSION_STRING;
static const char doc[] = "FastCarD: Fast Carrier Detection\n\n"
    "Takes a stream of raw 8-bit IQ samples from a RTL-SDR, splits it into "
    "fixed-sized blocks, and, if a carrier is detected in a block, outputs the "
    "block ID, timestamp and the block's raw samples encoded in base64.";

fargs_t* args;
fastcard_t* fastcard = NULL;
char* output_file = NULL;

static error_t parse_opt (int key, char *arg, struct argp_state *state) {
    if (key == 'o') {
        output_file = arg;
        return 0;
    } else if (key == ARGP_KEY_ARG) {
        // We don't take any arguments
        argp_usage(state);
    }

    int result = fargs_parse_opt(args, key, arg);
    if (result == FARGS_UNKNOWN) {
        return ARGP_ERR_UNKNOWN;
    } else if (result == FARGS_INVALID_VALUE) {
        argp_usage(state);
    }

    return 0;
}

void signal_handler(int signo) {
    (void)signo;  // unused
    fastcard_cancel(fastcard);
}

static struct argp_option extra_options[] = {
    {"output", 'o', "<FILE>", 0,
        "Output card file ('-' for stdout)\n[default: no output]", 1}
};
#define NUM_EXTRA_OPTIONS 1


int main(int argc, char **argv) {
    struct argp_option options[FARGS_NUM_OPTIONS + NUM_EXTRA_OPTIONS];
    memcpy(options,
           extra_options,
           sizeof(struct argp_option)*NUM_EXTRA_OPTIONS);
    memcpy(options + NUM_EXTRA_OPTIONS,
           fargs_options,
           sizeof(struct argp_option)*FARGS_NUM_OPTIONS);
    struct argp argp = {options, parse_opt, NULL,
                        doc, NULL, NULL, NULL};

    //// Set the stage
    args = fargs_new();
    argp_parse(&argp, argc, argv, 0, 0, 0);

    // variables
    FILE *out = NULL;
    FILE *info = NULL;
    char *base64 = NULL;
    int exit_code = 0;

    // open streams
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

    if (args->silent) {
        info = NULL;
    } else if (out == stdout) {
        info = stderr;
    } else {
        info = stdout;
    }

    // init stuff
    fastcard = fastcard_new(args);
    if (fastcard == NULL) {
        exit_code = -1;
        goto free;   
    }

    base64 = (char*) malloc((2*args->block_len+2)/3*4 + 1);
    if (base64 == NULL) {
        exit_code = -1;
        goto free;
    }

    bool sdr_input = false;
    if (args->input_file) {
        sdr_input = (strcmp(args->input_file, "rtlsdr") == 0);   
    }
    if (info != NULL) {
        fargs_print_summary(args, info, sdr_input);
        fflush(info);
    }
    if (out != NULL && out != stdout) {
        fargs_print_card_header(args, out, sdr_input, argp_program_version);
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGQUIT, signal_handler);
    signal(SIGPIPE, signal_handler);

    //// Start!
    exit_code = fastcard_start(fastcard);
    if (exit_code != 0) goto free;

    if (args->skip > 0 && info != NULL) {
        fprintf(info, "\nSkipping %u block(s)... ", args->skip);
        fflush(info);
    }
    unsigned skip = args->skip;
    unsigned cnt = 0;

    const fastcard_data_t* data;
    int ret = 0;
    while (true) {
        ret = fastcard_next(fastcard, &data);
        if (ret != 0) {
            break;
        }

        if (skip > 0) {
            --skip;
            if (skip == 0 && info != NULL) {
                fprintf(info, "done\n\n");
                fflush(info);
            }
            continue;
        }

        if (data->detected) {
            const block_t* block = data->block;
            const cardet_detection_t* det = &data->detection;
            if (info != NULL) {
                fprintf(info,
                        "block #%ld: mag[%d] = %.1f "
                        "(thresh = %.1f, noise = %.1f)\n",
                         data->block->index,
                         det->argmax, sqrt(det->max),
                         sqrt(det->threshold), sqrt(det->noise));
            }

            if (out != NULL) {
                Base64encode(base64,
                             (const char*) block->raw_samples,
                             args->block_len * 2);
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

    if (info != NULL) {
        fprintf(info, "\nRead %d blocks.\n", cnt);
        fastcard_print_stats(fastcard, info);
    }

    if (ret != 1) {
        // reader didn't stop gracefully
        exit_code = ret;
    }


    //// Free stuff
free:
    if (fastcard) {
        fastcard_free(fastcard);
    }
    if (base64) {
        free(base64);
    }
    if (info != NULL) {
        fflush(info);
    }
    if (out != NULL) {
        fflush(out);
    }
    if (out != NULL && out != stdout) {
        fclose(out);
    }
    if (args != NULL) {
        free(args);
    }

    return exit_code;
}
