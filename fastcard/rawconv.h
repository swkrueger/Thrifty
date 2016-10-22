// Convert raw 8-bit I/Q samples from an RTL-SDR to an array of complex values.

#ifndef RAWCONV_H
#define RAWCONV_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <complex.h>
#include <stdint.h>
#include <stdlib.h>

#define RAWCONV_ZERO 127

typedef struct {
    complex float lut[0x10000];
} rawconv_t;

void rawconv_init(rawconv_t *rawconv);
void rawconv_to_complex(rawconv_t *rawconv,
                        complex float* output,
                        uint16_t* input,
                        size_t len);

#ifdef __cplusplus
}
#endif

#endif /* RAWCONV_H */
