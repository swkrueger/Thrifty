// Perform carrier detection

#ifndef CARDET_H
#define CARDET_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

typedef struct {
    float threshold_constant;
    float threshold_snr;
    int carrier_freq_min;
    int carrier_freq_max;
    size_t fft_len;
} cardet_settings_t;

typedef struct {
    uint16_t argmax;
    float max;
    float threshold;
    float noise;
    float fft_sum;
} cardet_detection_t;

bool cardet_detect(cardet_settings_t *settings,
                   cardet_detection_t *output,
                   float *fft_power);

int cardet_normalize_window(cardet_settings_t *settings);

#ifdef __cplusplus
}
#endif

#endif /* CARDET_H */
