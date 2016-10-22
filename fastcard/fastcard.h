// Fast carrier detector API

#ifndef FASTFARD_H
#define FASTFARD_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <complex.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include "cardet.h"
#include "rawconv.h"
#include "reader.h"
#include "fft.h"

#include "fargs_type.h"

typedef struct {
    block_t* block;
    complex float* samples;  // owned by samples_to_fft
    complex float* fft;      // owned by samples_to_fft
    float* fft_power;
    bool detected;
    cardet_detection_t detection;
} fastcard_data_t;

typedef struct {
    fastcard_data_t data;

    // submodules
    reader_t* reader;
    fft_state_t* samples_to_fft;
    rawconv_t rawconv;
    cardet_settings_t cardet_settings;

    bool volatile keep_running;
    FILE* in;
    fargs_t* args;
} fastcard_t;

fastcard_t* fastcard_new(fargs_t* args, FILE* in);
void fastcard_free(fastcard_t* fc);
int fastcard_start(fastcard_t* fc);
int fastcard_next(fastcard_t* fc, const fastcard_data_t ** data);
void fastcard_cancel(fastcard_t* fc);
void fastcard_print_stats(fastcard_t* fc, FILE* out);

#ifdef __cplusplus
}
#endif

#endif /* FASTFARD_H */
