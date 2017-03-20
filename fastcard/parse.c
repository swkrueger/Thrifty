#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "parse.h"

// TODO: report error to user without printing to stderr

double parse_si_float(char *s) {
    // Copied from rtlsdr/src/convenience/convenience.c
    // <copy>

	char last;
	int len;
	double suff = 1.0;
	len = strlen(s);
	last = s[len-1];
	s[len-1] = '\0';
	switch (last) {
		case 'g':
		case 'G':
			suff *= 1e3;
		case 'm':
		case 'M':
			suff *= 1e3;
		case 'k':
		case 'K':
			suff *= 1e3;
			suff *= atof(s);
			s[len-1] = last;
			return suff;
	}
	s[len-1] = last;
	return atof(s);

    // </copy>
}

bool parse_carrier_str(const char *arg,
                       int *freq_min,
                       int *freq_max) {
    int r = sscanf(arg, "%d-%d", freq_min, freq_max);

    if (r == 1) {
        *freq_max = *freq_min;
    } else if (r != 2) {
        fprintf(stderr, "Argument '--carrier' contains an invalid value.\n");
        return false;
    }
    
    return true;
}

bool parse_theshold_str(const char *arg,
                        float *constant,
                        float *snr) {
    float f;
    int n;
    
    bool got_constant = false;
    bool got_snr = false;

    *constant = 0;
    *snr = 0;

    while (sscanf(arg, "%f%n", &f, &n) == 1) {
        arg += n;
        switch (*arg) {
            case 'c':
                arg += 1;
            case '\0':
                if (got_constant) {
                    fprintf(stderr, "Argument '--threshold' contains more than "
                                    "one value for constant.\n");
                    return false;
                }
                *constant = f;
                got_constant = true;
                break;
            case 's':
                if (got_snr) {
                    fprintf(stderr, "Argument '--threshold' contains more than "
                                    "one value for SNR.\n");
                    return false;
                }
                *snr = f;
                arg +=1;
                got_snr = true;
                break;
        }
    }

    if (*arg != '\0') {
        fprintf(stderr, "Argument '--threshold' contains an invalid value.\n");
        return false;
    }

    return true;
}
