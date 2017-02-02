#include <stdlib.h>
#include <string.h>

#include <volk/volk.h>

#include "raw_reader.h"
#include "card_reader.h"
#include "rtlsdr_reader.h"

#include "fastcard.h"


fastcard_t* fastcard_new(fargs_t* args) {
    if (args->history_len > args->block_len) {
        fprintf(stderr, "History length cannot be larger than block length.\n");
        return NULL;
    }

    FILE* in;
    if (strlen(args->input_file) == 0 || strcmp(args->input_file, "-") == 0) {
        in = stdin;
    } else if (strcmp(args->input_file, "rtlsdr") == 0) {
        in = NULL;
    } else {
        in = fopen(args->input_file, "rb");
        if (in == NULL) {
            perror("failed to open input file");
            return NULL;
        }
    }

    if (args->wisdom_file != NULL && strlen(args->wisdom_file) > 0) {
        int r = fftwf_import_wisdom_from_filename(args->wisdom_file);
        if (r == 0) {
            fprintf(stderr, "failed to import wisdom file\n");
        }
    }

    fastcard_t* fc = malloc(sizeof(fastcard_t));
    if (fc == NULL) {
        return NULL;
    }

    // ensure free will only free the variables that have been initialized
    fc->in = in;
    fc->args = args;
    fc->data.block = NULL;
    fc->reader = NULL;
    fc->samples_to_fft = NULL;
    fc->data.fft_power = NULL;
    fc->keep_running = true;

    // init stuff
    rawconv_init(&fc->rawconv);

    fc->data.block = reader_block_new(args->block_len);
    if (fc->data.block == NULL) {
        goto fail;
    }

    reader_settings_t reader_settings;
    reader_settings.output = fc->data.block;
    reader_settings.block_size = args->block_len;
    reader_settings.history_size = args->history_len;

    if (in == NULL) {
        rtlsdr_settings_t sdr_settings;
        sdr_settings.sample_rate = args->sdr_sample_rate;
        sdr_settings.gain = args->sdr_gain;
        sdr_settings.frequency = args->sdr_freq;
        sdr_settings.dev_index = args->sdr_dev_index;

        fc->reader = rtlsdr_reader_new(reader_settings, &sdr_settings);
        args->sdr_gain = sdr_settings.gain;  // get exact gain
    } else {
        if (args->input_card) {
            fc->reader = card_reader_new(reader_settings, in);
            // don't skip blocks when reading from .card file
            args->skip = 0;
        } else {
            fc->reader = raw_reader_new(reader_settings, in);
        }
    }
    if (fc->reader == NULL) {
        goto fail;
    }
    fc->samples_to_fft = fft_new(args->block_len, true);
    fc->data.samples = fft_get_input(fc->samples_to_fft);
    fc->data.fft = fft_get_output(fc->samples_to_fft);
    size_t alignment = volk_get_alignment();
    fc->data.fft_power = (float*) volk_malloc(args->block_len * sizeof(float),
                                              alignment);
    if (fc->data.fft_power == NULL) {
        goto fail;
    }

    fc->cardet_settings.threshold_constant = args->threshold_const;
    fc->cardet_settings.threshold_snr = args->threshold_snr;
    fc->cardet_settings.carrier_freq_min = args->carrier_freq_min;
    fc->cardet_settings.carrier_freq_max = args->carrier_freq_max;
    fc->cardet_settings.fft_len = args->block_len;
    if (cardet_normalize_window(&fc->cardet_settings) != 0) {
        goto fail;
    }
    args->carrier_freq_min = fc->cardet_settings.carrier_freq_min;
    args->carrier_freq_max = fc->cardet_settings.carrier_freq_max;

    if (args->skip > 0) {
        fc->data.block->index = ((int64_t)args->skip)*-1 - 1;
    }

    return fc;

fail:
    fastcard_free(fc);
    return NULL;
}

void fastcard_free(fastcard_t* fc) {
    if (fc->in && fc->in != stdin) {
        fclose(fc->in);
    }
    reader_block_free(fc->data.block);
    reader_free(fc->reader);
    fft_free(fc->samples_to_fft);
    free(fc->data.fft_power);
    free(fc);
}

int fastcard_start(fastcard_t* fc) {
    reader_start(fc->reader);
    return 0;
}

static void fastcard_stop(fastcard_t* fc) {
    reader_stop(fc->reader);

    // TODO: move this somewhere else
    if (fc->args->wisdom_file != NULL && strlen(fc->args->wisdom_file) > 0) {
        int r = fftwf_export_wisdom_to_filename(fc->args->wisdom_file);
        if (r == 0) {
            fprintf(stderr, "failed to export wisdom file\n");
        }
    }
}

// Read the next block of data
int fastcard_next(fastcard_t* fc) {
    if (!fc->keep_running) {
        fastcard_stop(fc);
        return 1;
    }

    int ret = reader_next(fc->reader);

    if (ret != 0) {
        if (ret != 1) {
            fprintf(stderr, "reader_next() failed\n");
        }
        fastcard_stop(fc);
        return ret;
    }

    fastcard_data_t* d = &fc->data;

    // TODO: move rawconv to reader
    rawconv_to_complex(&fc->rawconv,
                       d->samples,
                       fc->data.block->raw_samples,
                       fc->args->block_len);

    return 0;
}

// Process the last block of data.
// Should be called after fastcard_next.
int fastcard_process(fastcard_t* fc, const fastcard_data_t ** data) {
    fastcard_data_t* d = &fc->data;
    fft_perform(fc->samples_to_fft);
    volk_32fc_magnitude_squared_32f_a(d->fft_power,
                                      (lv_32fc_t*)d->fft,
                                      fc->args->block_len);
    d->detected = cardet_detect(&fc->cardet_settings,
                                &d->detection,
                                d->fft_power);

    *data = d;
    return 0;
}

// Convenience function for reading and processing the next block of data.
int fastcard_process_next(fastcard_t* fc, const fastcard_data_t ** data) {
    int ret = fastcard_next(fc);
    if (ret != 0) {
        return ret;
    }
    ret = fastcard_process(fc, data);
    return ret;
}

void fastcard_cancel(fastcard_t* fc) {
    fc->keep_running = false;
    reader_cancel(fc->reader);
}

void fastcard_print_stats(fastcard_t* fc, FILE* out) {
    if (fc->in == NULL) {
        // print SDR buffer histogram
        rtlsdr_reader_print_histogram(fc->reader, out);
    }
}
