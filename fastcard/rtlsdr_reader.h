// Read from RTL-SDR by means of librtlsdr

#ifndef RTLSDR_READER_H
#define RTLSDR_READER_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>
#include <stdbool.h>

#include "reader.h"

typedef struct {
    uint32_t frequency;
    uint32_t sample_rate;
    uint16_t gain;
    uint8_t dev_index;
} rtlsdr_settings_t;

reader_t * rtlsdr_reader_new(reader_settings_t reader_settings,
                             rtlsdr_settings_t *rtl_settings);

void rtlsdr_reader_print_histogram(reader_t* reader, FILE* output);

#ifdef LIBRTLSDR_BIAS_TEE_SUPPORT
int rtlsdr_reader_set_bias_tee(reader_t* reader, bool on);
#endif /* LIBRTLSDR_BIAS_TEE_SUPPORT */

void rtlsdr_reader_set_standby(reader_t* reader, bool on);

#ifdef __cplusplus
}
#endif

#endif /* RTLSDR_READER_H */
