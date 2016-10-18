#include <stdint.h>
#include <stdio.h>
#include <volk/volk.h>

#include "cardet.h"

bool cardet_detect(cardet_settings_t *settings,
                   cardet_detection_t *output,
                   float *fft_power) {

    float sum = 0;
    volk_32f_accumulator_s32f(&sum, fft_power, settings->fft_len);

    uint16_t argmax; // todo: volk_malloc
    volk_32f_index_max_16u(
            &argmax,
            fft_power + settings->carrier_freq_min,
            settings->carrier_freq_max - settings->carrier_freq_min + 1);
    argmax += settings->carrier_freq_min;
    float max = fft_power[argmax];

    float noise_power = 0;
    if (sum != 0) {
        noise_power = (sum - 2*max) / (settings->fft_len - 1);
    }
    float threshold = settings->threshold_constant + \
                      settings->threshold_snr * noise_power;

    if (max > threshold) {
        if (output != NULL) {
            output->argmax = argmax;
            output->max = max;
            output->threshold = threshold;
            output->noise = noise_power;
            output->fft_sum = sum;
        }
        return true;
    }

    return false;
}

int cardet_normalize_window(cardet_settings_t *settings) {
    if (settings->carrier_freq_min < 0 && settings->carrier_freq_max >= 0) {
        // TODO: don't print in library: report error string to caller
        fprintf(stderr, "Carrier frequency window range not supported.\n");
        return -1;
    }
    if (settings->carrier_freq_min < 0) {
        settings->carrier_freq_min = (settings->fft_len +
                                      settings->carrier_freq_min);
    }
    if (settings->carrier_freq_max < 0) {
        settings->carrier_freq_max = settings->fft_len +
                                     settings->carrier_freq_max;
    }
    if ((size_t)settings->carrier_freq_min >= settings->fft_len
            || (size_t)settings->carrier_freq_max >= settings->fft_len) {
        // TODO: don't print in library: report error string to caller
        fprintf(stderr, "Carrier frequency window out of range.\n");
        return -1;
    }
    if (settings->carrier_freq_max < settings->carrier_freq_min) {
        int t = settings->carrier_freq_max;
        settings->carrier_freq_max = settings->carrier_freq_min;
        settings->carrier_freq_min = t;
    }
    return 0;
}
