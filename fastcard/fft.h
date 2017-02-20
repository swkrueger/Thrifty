// Common interface for calculating FFT using different backends.
//
// Only FFTW is currently supported as backend, but GPU-FFT will be (re-)added
// in the future if necessary.

#ifndef FFT_H
#define FFT_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <fftw3.h>

typedef struct {
    float real;
    float imag;
} fcomplex;

typedef struct {
    fftwf_plan plan;
    fcomplex *input;
    fcomplex *output;
    size_t fft_len;
    bool forward;
} fft_state_t;

fft_state_t * fft_new(size_t fft_len,
                      bool forward);
void fft_free(fft_state_t *state);
void fft_perform(fft_state_t *state);

fcomplex * fft_get_input(fft_state_t *state);
fcomplex * fft_get_output(fft_state_t *state);

#ifdef __cplusplus
}
#endif

#endif /* FFT_H */
