#include <endian.h>

#include "rawconv.h"

void rawconv_init(rawconv_t *rawconv) {
    complex float *lut = rawconv->lut;
    // generate lookup table for raw-to-complex conversion
    for (size_t i = 0; i <= 0xffff; ++i) {
#if __BYTE_ORDER == __LITTLE_ENDIAN
        ((float*)&lut[i])[0] = ((float)(i & 0xff) - 127.4f) * (1.0f/128.0f);
        ((float*)&lut[i])[1] = ((float)(i >> 8) - 127.4f) * (1.0f/128.0f);
#elif __BYTE_ORDER == __BIG_ENDIAN
        ((float*)&lut[i])[0] = ((float)(i >> 8) - 127.4f) * (1.0f/128.0f);
        ((float*)&lut[i])[1] = ((float)(i & 0xff) - 127.4f) * (1.0f/128.0f);
#else
#error "Could not determine endianness"
#endif
    }
}

void rawconv_to_complex(rawconv_t *rawconv,
                        complex float* output,
                        uint16_t* input,
                        size_t len) {
    for (size_t i = 0; i < len; ++i) {
        output[i] = rawconv->lut[input[i]];
    }
}
