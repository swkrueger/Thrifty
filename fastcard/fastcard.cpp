#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex>

typedef std::complex<float> fc_complex;

// Settings
int block_size_log2N = 13; // 8096
int history_size = 2085;

// Internal
int block_size = 1<<block_size_log2N;

// Buffers
uint16_t *raw_samples;
size_t raw_samples_len;

void init_buffers() {
    raw_samples = (uint16_t*) calloc(block_size, sizeof(uint16_t));
    raw_samples_len = 0;
}

bool read_next_block(FILE *f) {
    // copy history
    size_t b = block_size - history_size;
    memcpy(raw_samples,
           raw_samples + b,
           history_size * 2);

    // read new data
    size_t c = fread(raw_samples + history_size, 2, b, f);

    if (c != b) {
        if (!feof(f)) {
            perror("Short read");
        }
        return false;
    }
    return true;
}

void free_buffers() {
    free(raw_samples);
}

int main() {
    FILE* in = fopen("test.dat", "rb");
    if (in == NULL) {
        perror("Failed to open input file");
        exit(1);
    }

    init_buffers();

    while (read_next_block(in)) {
        for (int i = 0; i < block_size; ++i) {
            printf("%04x ", raw_samples[i]);
        }
        printf("\n---------------------------\n");
    }

    if (in != stdin) {
        fclose(in);
    }
}
