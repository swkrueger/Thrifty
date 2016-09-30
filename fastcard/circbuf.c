#include <stdlib.h>
#include <string.h>

#include "circbuf.h"

circbuf_t* circbuf_new(size_t size) {
    circbuf_t *circbuf;
    circbuf = malloc(sizeof(circbuf_t));
    if (circbuf == NULL || !circbuf_init(circbuf, size)) {
        return NULL;
    } else {
        return circbuf;
    }
}

bool circbuf_init(circbuf_t* circbuf, size_t size) {
    if (pthread_mutex_init(&circbuf->mutex, NULL) != 0) {
        goto fail;
    }
    if (pthread_cond_init(&circbuf->can_produce, NULL) != 0) {
        goto fail;
    }
    if (pthread_cond_init(&circbuf->can_consume, NULL) != 0) {
        goto fail;
    }

    circbuf->size = size;
    circbuf->len = 0;
    circbuf->head = 0;
    circbuf->tail = 0;
    circbuf->cancel = false;
    circbuf->num_overflows = 0;

    circbuf->histogram = malloc(CIRCBUF_HISTOGRAM_LEN * sizeof(unsigned));
    if (circbuf->histogram == NULL) {
        goto fail;
    }
    for (int i = 0; i < CIRCBUF_HISTOGRAM_LEN; ++i) circbuf->histogram[i] = 0;

    circbuf->buf = malloc(circbuf->size);
    if (circbuf->buf == NULL) {
        goto fail;
    }

    return true;

fail:
    // TODO: destroy mutex, conds and buffers
    return false;
}

void circbuf_free(circbuf_t* circbuf) {
    pthread_mutex_destroy(&circbuf->mutex);
    pthread_cond_destroy(&circbuf->can_produce);
    pthread_cond_destroy(&circbuf->can_consume);
    if (circbuf->histogram != NULL)
        free(circbuf->histogram);
    if (circbuf->buf != NULL)
        free(circbuf->buf);

    free(circbuf);
}

bool circbuf_get(circbuf_t* circbuf, char* dest, size_t len) {
    pthread_mutex_lock(&circbuf->mutex);

    if (len > circbuf->size) {
        goto fail;
    }

    // wait for producer on underflow
    while (!circbuf->cancel && circbuf->len < len) {
        pthread_cond_wait(&circbuf->can_consume, &circbuf->mutex);
    }

    if (circbuf->cancel) {
        goto fail;
    }

    // consume data
    size_t len_to_end = circbuf->size - circbuf->tail;
    if (len_to_end > len) {
        len_to_end = len;
    }
    memcpy(dest, circbuf->buf + circbuf->tail, len_to_end);
    memcpy(dest + len_to_end, circbuf->buf, len - len_to_end);

    circbuf->len -= len;
    circbuf->tail += len;
    if (circbuf->tail >= circbuf->size) {
        circbuf->tail -= circbuf->size;
    }

    // inform a waiting producer that we've consumed data
    pthread_cond_signal(&circbuf->can_produce);

    pthread_mutex_unlock(&circbuf->mutex);
    return true;

fail:
    pthread_mutex_unlock(&circbuf->mutex);
    return false;
}

bool circbuf_put(circbuf_t* circbuf, char* src, size_t len) {
    pthread_mutex_lock(&circbuf->mutex);

    if (len > circbuf->size) {
        goto fail;
    }

    if (circbuf->len < circbuf->size) {
        circbuf->histogram[circbuf->len*CIRCBUF_HISTOGRAM_LEN/circbuf->size]++;
    }
    // wait for consumer on overflow
    if (circbuf->len + len >= circbuf->size) {
        circbuf->num_overflows++;
    }
    while (!circbuf->cancel && circbuf->len + len >= circbuf->size) {
        pthread_cond_wait(&circbuf->can_produce, &circbuf->mutex);
    }
    if (circbuf->cancel) {
        goto fail;
    }

    // add new data
    size_t len_to_end = circbuf->size - circbuf->head;
    if (len_to_end > len) {
        len_to_end = len;
    }

    memcpy(circbuf->buf + circbuf->head, src, len_to_end);
    memcpy(circbuf->buf, src + len_to_end, len - len_to_end);

    circbuf->len += len;
    circbuf->head += len;
    if (circbuf->head >= circbuf->size) {
        circbuf->head -= circbuf->size;
    }

    // inform a waiting consumer that new data is available
    pthread_cond_signal(&circbuf->can_consume);

    pthread_mutex_unlock(&circbuf->mutex);

    return true;

fail:
    pthread_mutex_unlock(&circbuf->mutex);
    return false;
}

void circbuf_cancel(circbuf_t* circbuf) {
    if (circbuf == NULL) {
        return;
    }

    pthread_mutex_lock(&circbuf->mutex);
    if (circbuf->cancel) return;
    circbuf->cancel = true;
    pthread_cond_signal(&circbuf->can_produce);
    pthread_cond_signal(&circbuf->can_consume);
    pthread_mutex_unlock(&circbuf->mutex);
}

unsigned* circbuf_histogram(circbuf_t* circbuf) {
    // TODO: lock and copy to buffer
    return circbuf->histogram;
}

unsigned circbuf_overflows(circbuf_t* circbuf) {
    // TODO: lock
    return circbuf->num_overflows;
}
