// Char FIFO circular buffer
//
// A circular buffer for which the producer and consumer can operate on
// separate pthread threads.

#ifndef CIRCBUF_H
#define CIRCBUF_H

#include <pthread.h>
#include <stdbool.h>

#define CIRCBUF_HISTOGRAM_LEN 20

typedef struct {
    char *buf;                      // the buffer
    size_t size;                    // size of buffer
    size_t len;                     // number of bytes occupied in the buffer
    size_t head;                    // position of producer
    size_t tail;                    // position of consumer
    bool cancel;                    // cancel get and put operations
    unsigned *histogram;            // record buffer occupancy
    unsigned num_overflows;         // number of overflow events
    pthread_mutex_t mutex;          // protect circbuf_t data
    pthread_cond_t can_produce;     // signaled when items have been removed
    pthread_cond_t can_consume;     // signaled when items have been added
} circbuf_t;

/// Create a new circular buffer.
/// Returns NULL if an error occurred.
circbuf_t* circbuf_new(size_t size);

/// Allocate memory for a circular buffer.
/// Returns true on success.
bool circbuf_init(circbuf_t* circbuf, size_t size);

/// Deallocate memory of a circular buffer.
void circbuf_free(circbuf_t* circbuf);

/// Read exactly "len" bytes from the circular buffer.
/// Will wait for producer if enough data is not available.
bool circbuf_get(circbuf_t* circbuf, char* dest, size_t len);

/// Write exactly "len" bytes to the circular buffer.
/// Will increase overflow counter and wait for consumer if the buffer is full.
bool circbuf_put(circbuf_t* circbuf, char* src, size_t len);

/// Wait for the next circbuf_put (or circbuf_cancel) call.
bool circbuf_wait_put(circbuf_t* circbuf);

/// Clear the circular buffer and resume the producer if it is blocked.
void circbuf_clear(circbuf_t* circbuf);

/// Reset the state of the circular buffer to where it is after construction
/// Do not use this function when the buffer is still in use.
void circbuf_reset(circbuf_t* circbuf);

/// Return the number of overflow events that occurred.
unsigned circbuf_overflows(circbuf_t* circbuf);

/// Return the occupancy histogram
unsigned* circbuf_histogram(circbuf_t* circbuf);

/// Cancel get waiting for more data and put waiting for data to be consumed.
void circbuf_cancel(circbuf_t* circbuf);

#endif /* CIRCBUF_H */
