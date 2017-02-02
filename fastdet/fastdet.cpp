// A quick-n-dirty proof-of-concept fast C++ implementation of Thrifty detect.
// This is a mess. This should be refactored.

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <memory>

#include <signal.h>
#include <stdio.h>

#include <argp.h>

#include "corr_detector.h"
#include "parse.h"
#include "lib/base64.h"

using namespace std;


//// CLI stuff
// TODO: use proper command-line parser (e.g. tclap)

const char *argp_program_version = "fastdet " VERSION_STRING;
static const char doc[] = "FastDet: Fast Detector\n\n"
    "Like Thrifty, but faster.";

#define NUM_EXTRA_OPTIONS 6
static struct argp_option extra_options[] = {
    {"output", 'o', "<FILE>", 0,
        "Output card file\n('-' for stdout)\n[default: no output]", 1},
    {"card-output", 'x', "<FILE>", 0,
     "Write block to card file on detect\n('-' for stdout)\n[default: no output]", 1},

    // Correlator
    {0, 0, 0, 0, "Correlator settings:", 5},
    {"corr-threshold", 'u', "<constant>c<snr>s", 0,
        "Correlation detection theshold\n[default: 15s]", 5},
    {"template", 'z', "<FILE>", 0,
        "Load template from a .tpl file\n[default: template.tpl]", 5},
    {"rxid", 'r', "<int>", 0,
        "This receiver's unique identifier\n[default: -1]", 5}
};

unique_ptr<fargs_t, decltype(free)*> args = {NULL, free};
std::string output_file;
std::string card_output_file;
std::string template_file = "template.tpl";
float arg_corr_thresh_const = 0;
float arg_corr_thresh_snr = 15;
int rxid = -1;

static error_t parse_opt (int key, char *arg, struct argp_state *state) {
    if (key == 'o') {
        output_file = arg;
    } else if (key == 'x') {
        card_output_file = arg;
    } else if (key == 'u') {
        if (!parse_theshold_str(arg,
                                &arg_corr_thresh_const,
                                &arg_corr_thresh_snr)) {
            argp_usage(state);
        }
    } else if (key == 'z') {
        template_file = arg;
    } else if (key == 'r') {
        rxid = atoi(arg);
    } else if (key == ARGP_KEY_ARG) {
        // We don't take any arguments
        argp_usage(state);
    } else {
        int result = fargs_parse_opt(args.get(), key, arg);
        if (result == FARGS_UNKNOWN) {
            return ARGP_ERR_UNKNOWN;
        } else if (result == FARGS_INVALID_VALUE) {
            argp_usage(state);
        }
    }

    return 0;
}

std::unique_ptr<CarrierDetector> carrier_det;

void signal_handler(int signo) {
    (void)signo;  // unused
    if (carrier_det) {
        carrier_det->cancel();
    }
}


int main(int argc, char **argv) {
    // Argument parsing mess
    struct argp_option options[FARGS_NUM_OPTIONS + NUM_EXTRA_OPTIONS];
    memcpy(options,
           extra_options,
           sizeof(struct argp_option)*NUM_EXTRA_OPTIONS);
    memcpy(options + NUM_EXTRA_OPTIONS,
           fargs_options,
           sizeof(struct argp_option)*FARGS_NUM_OPTIONS);
    struct argp argp = {options, parse_opt, NULL,
                        doc, NULL, NULL, NULL};

    args.reset(fargs_new());
    argp_parse(&argp, argc, argv, 0, 0, 0);

    try {
        CFile out = CFile(output_file);
        CFile card = CFile(card_output_file);
        CFile info;
        if (!args->silent) {
            info.open((out.file() == stdout) ? stderr : stdout);
        }

        carrier_det.reset(new CarrierDetector(args.get()));
        vector<float> template_samples = load_template(template_file);
        CorrDetector corr_detect(template_samples,
                                 args->block_len,
                                 args->history_len,
                                 arg_corr_thresh_const,
                                 arg_corr_thresh_snr);

        vector<char> base64((2*args->block_len+2)/3*4 + 10);

        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        signal(SIGQUIT, signal_handler);
        signal(SIGPIPE, signal_handler);

        // print header
        bool input_from_sdr = false;
        if (args->input_file) {
            input_from_sdr = (strcmp(args->input_file, "rtlsdr") == 0);
        }
        if (info.file() != NULL) {
            fargs_print_summary(args.get(), info.file(), input_from_sdr);
            info.printf("receiver id: %d\n", rxid);
            info.printf("corr threshold: constant = %g; snr = %g\n",
                       arg_corr_thresh_const, arg_corr_thresh_snr);
            info.printf("template: %s\n\n", template_file.c_str());
            info.flush();
        }

        if (card.file() != NULL && card.file() != stdout) {
            fargs_print_card_header(args.get(), card.file(),
                                    input_from_sdr, argp_program_version);
        }

        // Start detection!
        unsigned skip = args->skip;
        unsigned cnt = 0;
        if (skip > 0) {
            info.printf("\nSkipping %u block(s)... ", args->skip);
            info.flush();
        }

        carrier_det->start();

        while (carrier_det->process_next()) {
            if (skip > 0) {
                --skip;
                if (skip == 0) {
                    info.printf("done\n\n");
                    info.flush();
                }
                continue;
            }
            ++cnt;

            const fastcard_data_t& carrier = carrier_det->data();
            if (!carrier.detected) {
                continue;
            }

            // TODO: Don't block, but use a producer / consumer queue to
            // perform correlation detection async

            const CorrDetection corr = corr_detect.detect(carrier);

            int64_t block_idx = carrier.block->index;
            double soa = ((args->block_len - args->history_len) *
                          block_idx + corr.peak_idx) + corr.peak_offset;

            if (corr.detected) {
                // output toad
                if (out.file() != NULL) {
                    out.printf("%d %ld.%06ld %" PRId64 " %.8f"
                               " %u %.12f %f %f %u %f %f %f\n",
                               rxid,
                               carrier.block->timestamp.tv_sec,
                               carrier.block->timestamp.tv_usec,
                               carrier.block->index,
                               soa,
                               corr.peak_idx,
                               corr.peak_offset,
                               sqrt(corr.peak_power),
                               sqrt(corr.noise_power),
                               carrier.detection.argmax,
                               corr.carrier_offset,
                               sqrt(carrier.detection.max),
                               sqrt(carrier.detection.noise)
                               );
                }

                if (card.file() != NULL) {
                    Base64encode(base64.data(),
                                 (const char*) carrier.block->raw_samples,
                                 args->block_len * 2);
                    card.printf("%ld.%06ld %" PRId64 " %s\n",
                                carrier.block->timestamp.tv_sec,
                                carrier.block->timestamp.tv_usec,
                                carrier.block->index,
                                base64.data());
                }
            }

            if (info.file() != NULL) {
                float carrier_snr_db = 10 * log10(carrier.detection.max /
                                                  carrier.detection.noise);

                info.printf("block #%" PRId64 ": carrier @ %3u %+.1f = "
                            "%4.0f / %2.0f [>%2.0f] = %2.0f dB",
                            block_idx,
                            carrier.detection.argmax,
                            corr.carrier_offset,
                            sqrt(carrier.detection.max),
                            sqrt(carrier.detection.noise),
                            sqrt(carrier.detection.threshold),
                            carrier_snr_db);

                if (corr.detected) {
                    float corr_snr_db = 10 * log10(corr.peak_power /
                                                   corr.noise_power);
                    info.printf("; corr = %4.0f / %2.0f [>%2.0f] = %2.0f dB",
                                sqrt(corr.peak_power),
                                sqrt(corr.noise_power),
                                sqrt(corr.threshold),
                                corr_snr_db);
                }

                info.printf("\n");
            }
        }

        if (info.file() != NULL) {
            info.printf("\nRead %d blocks.\n", cnt);
            carrier_det->print_stats(info.file());
        }

    } catch (FastcardException& e) {
        cerr << e.what() << endl;
        return e.getCode();
    } catch (std::exception& e) {
        cerr << e.what() << endl;
        return -1;
    }

    return 0;
}
