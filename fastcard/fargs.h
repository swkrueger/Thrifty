// Parse Fastcard (CLI) arguments ("fargs") using argp

#ifndef FARGS_H
#define FARGS_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#include <argp.h>

typedef struct {
    size_t block_len;
    size_t history_len;
    
    float threshold_const;
    float threshold_snr;
    int carrier_freq_min;
    int carrier_freq_max;
    unsigned skip;
    
    char *input_file;
    char *output_file;
    bool input_card;

    uint32_t sdr_freq;
    uint32_t sdr_sample_rate;
    int sdr_gain;
    uint32_t sdr_dev_index;

    bool silent;
} fargs_t;

extern const struct argp_option fargs_options[];
fargs_t* fargs_new();
error_t fargs_parse_opt(fargs_t *fargs,
                        int key,
                        char *arg,
                        struct argp_state *state);
int fargs_open_streams(fargs_t *fa, FILE** in, FILE** out);

// TODO: move print functions to another module
void fargs_print_summary(fargs_t *fa, FILE* out, bool sdr);
void fargs_print_card_header(fargs_t *fa,
                             FILE* out,
                             bool sdr,
                             const char* tool);

#ifdef __cplusplus
}
#endif

#endif /* FARGS_H */
