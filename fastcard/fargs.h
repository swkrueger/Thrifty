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

#include "fargs_type.h"

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
