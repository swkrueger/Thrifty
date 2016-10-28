#include <stdlib.h>
#include <assert.h>

#include "fft.h"

fft_state_t * fft_new(size_t fft_len,
                      bool forward) {
    fft_state_t* state;
    state = malloc(sizeof(fft_state_t));
    if (state == NULL) {
        return NULL;
    }

    state->plan = NULL;
    state->fft_len = fft_len;

    assert(sizeof(fcomplex) == sizeof(fftwf_complex));

    size_t num_bytes = state->fft_len * sizeof(fcomplex);
    state->input = (fcomplex*) fftwf_malloc(num_bytes);
    state->output = (fcomplex*) fftwf_malloc(num_bytes);

    if (state->input == NULL || state->output == NULL) {
        // fprintf(stderr, "init fft malloc failed\n");
        goto fail;
    }

    // TODO: load and save wisdom file

    state->plan = fftwf_plan_dft_1d(state->fft_len,
                                    (fftwf_complex*) state->input,
                                    (fftwf_complex*) state->output,
                                    forward ? FFTW_FORWARD : FFTW_BACKWARD,
                                    FFTW_MEASURE);
    if (state->plan == NULL) {
        // fprintf(stderr, "failed to create fft plan\n");
        goto fail;
    }

    return state;

fail:
    fft_free(state);
    return NULL;
}

void fft_free(fft_state_t *state) {
    if (state == NULL) {
        return;
    }
    if (state->plan != NULL) {
        fftwf_destroy_plan(state->plan);
    }
    if (state->input != NULL) {
        fftwf_free(state->input);
    }
    if (state->output != NULL) {
        fftwf_free(state->output);
    }
    free(state);
}

void fft_perform(fft_state_t *state) {
    fftwf_execute(state->plan);
}

fcomplex * fft_get_input(fft_state_t *state) {
    return state->input;
}
fcomplex * fft_get_output(fft_state_t *state) {
    return state->output;
}
