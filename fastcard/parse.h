// Helper functions for parsing command-line arguments

#ifndef PARSE_H
#define PARSE_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>

// Parse float with SI suffix (kilo, mega, giga)
double parse_si_float(char *s);
bool parse_carrier_str(const char *arg,
                       int *freq_min,
                       int *freq_max);
bool parse_theshold_str(const char *arg,
                        float *constant,
                        float *snr);

#ifdef __cplusplus
}
#endif

#endif /* PARSE_H */
