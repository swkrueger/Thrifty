#include <stdlib.h>
#include <string.h>

#include "raw_reader.h"

typedef struct {
    FILE* file;
    reader_settings_t settings;
} raw_reader_t;

void raw_reader_free(raw_reader_t* state) {
    free(state);  // will handle state == NULL
}

int raw_reader_next(raw_reader_t* state) {
    // copy history
    block_t* output = state->settings.output;
    size_t history_size = state->settings.history_size;

    size_t new_len = state->settings.block_size - history_size;

    memcpy(output->raw_samples,
           output->raw_samples + new_len,
           history_size * 2);

    // Read new data
    size_t read = fread(output->raw_samples + history_size,
                        2,
                        new_len,
                        state->file);


    if (read != new_len) {
        if (!feof(state->file)) {
            perror("Short read");
        }
        return -1;
    }

    // Capture metadata
    output->index++;
    gettimeofday(&output->timestamp, NULL);

    return 0;
}

reader_t * raw_reader_new(reader_settings_t settings,
                          FILE* file) {
    raw_reader_t* state;
    state = malloc(sizeof(raw_reader_t));
    reader_t* reader = malloc(sizeof(reader_t));
    if (state == NULL || reader == NULL) {
        free(state);
        free(reader);
        return NULL;
    }
    state->settings = settings;
    state->file = file;

    reader->context = state;
    reader->next = (reader_func_t)&raw_reader_next;
    reader->start = NULL;
    reader->stop = NULL;
    reader->cancel = NULL;
    reader->free = (reader_func_void_t)&raw_reader_free;

    return reader;
}
