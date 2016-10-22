#include <stdlib.h>
#include <string.h>

#include "card_reader.h"
#include "lib/base64.h"

typedef struct {
    FILE* file;
    char* base64;
    size_t base64_len;
    reader_settings_t settings;
} card_reader_t;

void card_reader_free(card_reader_t* state) {
    free(state);  // will handle state == NULL
}

int card_reader_next(card_reader_t* state) {
    // FIXME: don't write to stderr

    // copy history
    block_t* output = state->settings.output;
    size_t history_size = state->settings.history_size;

    size_t new_len = state->settings.block_size - history_size;

    memcpy(output->raw_samples,
           output->raw_samples + new_len,
           history_size * 2);

    // Read new data
    char c;
    int read;
    do {
        read = fscanf(state->file, "#%*[^\n]%c", &c);
    } while (read && !feof(state->file));
    read = fscanf(state->file,
                      " %ld.%ld %ld ",
                      &output->timestamp.tv_sec,
                      &output->timestamp.tv_usec,
                      &output->index);
    if (read != 3) {
        if (feof(state->file)) {
            return -1;
        }
        fprintf(stderr, "card_reader: failed to read metadata\n");
        return -2;
    }
    char* ret = fgets(state->base64, state->base64_len + 2, state->file);
    if (ret == NULL) {
        fprintf(stderr, "card_reader: failed to read base64 data\n");
        return -3;
    }
    size_t len = strlen(ret);
    if (len < state->base64_len + 1) {
        fprintf(stderr, "card_reader: line too short\n");
        return -4;
    }
    if (ret[len-1] != '\n') {
        fprintf(stderr, "card_reader: line too long\n");
        return -5;
    }

    // fgets will terminate string
    unsigned long num = Base64decode((char*)output->raw_samples, state->base64);
    if (num != 2*state->settings.block_size) {
        fprintf(stderr, "card_reader: block length is %ld, expected %ld\n",
                num,
                2*state->settings.block_size);
        return -6;
    }

    return 0;
}

reader_t * card_reader_new(reader_settings_t settings,
                          FILE* file) {
    card_reader_t* state;
    state = malloc(sizeof(card_reader_t));
    state->base64 = NULL;
    reader_t* reader = malloc(sizeof(reader_t));
    if (state == NULL || reader == NULL) {
        goto fail;
    }
    state->settings = settings;
    state->file = file;
    state->base64_len = (2*settings.block_size+2)/3*4;
    // base64 buffer: leave space for \n and \0
    state->base64 = (char*) malloc(state->base64_len + 2);
    if (state->base64 == NULL) {
        goto fail;
    }

    reader->context = state;
    reader->next = (reader_func_t)&card_reader_next;
    reader->start = NULL;
    reader->stop = NULL;
    reader->cancel = NULL;
    reader->free = (reader_func_void_t)&card_reader_free;

    return reader;

fail:
    if (state != NULL) {
        free(state->base64);
    }
    free(state);
    free(reader);
    return NULL;
}
