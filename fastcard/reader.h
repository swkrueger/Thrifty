// Common interface for all readers

#ifndef READER_H
#define READER_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include <stddef.h>
#include <sys/time.h>

typedef struct {
    struct timeval timestamp;
    int64_t index;
    uint16_t *raw_samples;
} block_t;

typedef struct {
    block_t* output;
    size_t block_size;
    size_t history_size;
} reader_settings_t;

typedef struct reader_t reader_t;
typedef int (*reader_func_t)(void* context);
typedef void (*reader_func_void_t)(void* context);
struct reader_t {
    void* context;
    reader_func_t next;
    reader_func_t start;
    reader_func_t stop;
    reader_func_void_t cancel;
    reader_func_void_t free;
};

int reader_next(reader_t* reader);
int reader_start(reader_t* reader);
int reader_stop(reader_t* reader);
void reader_cancel(reader_t* reader);
void reader_free(reader_t* reader);

block_t * reader_block_new(size_t len);  // new "clean" block
void reader_block_free(block_t *block);

#ifdef __cplusplus
}
#endif

#endif /* READER_H */
