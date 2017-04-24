#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <rtl-sdr.h>

#include "circbuf.h"

#include "rtlsdr_reader.h"

#define CIRCBUF_SIZE (16 * 16384 * 128)  // 32 MiB
#define RTLSDR_BUF_LENGTH (16 * 16384)   // 256 KiB
#define RTLSDR_BUF_NUM (16)              // \_> * 16 = 4 MiB

// TODO: replace byte-level circbuf with a block-level circular buffer
//       e.g. one block of data per slot instead of one byte per slot
//       (will have less overhead and will be simpler)

typedef struct {
    reader_settings_t settings;
    rtlsdr_dev_t *sdr;
    circbuf_t* circbuf;
    pthread_t sdr_thread;
    // WARNING: sdr_running and standby are accessed by multiple threads without a mutex.
    // FIXME: potential race condition
    bool volatile sdr_running;
    bool volatile standby;
    bool cancelled;
    int return_code;

    uint8_t* wip_block;        // incomplete block of data
    size_t wip_block_len;      // current occupancy of wip_block
    size_t wip_block_capacity; // number of new bytes per block
                               // (max size of wip_block)
    uint8_t* reader_block;     // temporary buffer for reader
} rtlsdr_reader_t;

// Copied from rtlsdr/src/convenience/convenience.c
// <copy>
static int nearest_gain(rtlsdr_dev_t *dev, int target_gain) {
    int i, err1, err2, count, nearest;
    int* gains;
    count = rtlsdr_get_tuner_gains(dev, NULL);
    if (count <= 0) {
        return 0;
    }
    gains = malloc(sizeof(int) * count);
    count = rtlsdr_get_tuner_gains(dev, gains);
    nearest = gains[0];
    for (i=0; i<count; i++) {
        err1 = abs(target_gain - nearest);
        err2 = abs(target_gain - gains[i]);
        if (err2 < err1) {
            nearest = gains[i];
        }
    }
    free(gains);
    return nearest;
}
// <copy>

static void sdr_callback(unsigned char *buf, uint32_t len, void *ctx) {
    if (ctx) {
        rtlsdr_reader_t* state = (rtlsdr_reader_t*) ctx;
        if (!state->sdr_running) return;

        bool have_timestamp = false;

        if (state->standby) {
            // Clear buffers
            state->wip_block_len = 0;
            // Do not consume anything
            return;
        }

        while (len > 0 && !state->standby) {
            size_t count = state->wip_block_capacity - state->wip_block_len;
            if (len < count) {
                count = len;
            }
            memcpy(state->wip_block + state->wip_block_len, buf, count);
            state->wip_block_len += count;
            len -= count;
            buf += count;

            if (state->wip_block_len == state->wip_block_capacity) {
                // block is full, flush it to the circular buffer
                // with a timestamp

                // get timestamp, append it to the block
                // if len > wip_block_capacity, reuse the timestamp from
                // the previous block
                if (!have_timestamp) {
                    have_timestamp = true;
                    struct timeval *timestamp = (struct timeval*)(
                                                state->wip_block +
                                                state->wip_block_capacity);
                    gettimeofday(timestamp, NULL);
                }

                circbuf_put(state->circbuf,
                            (char*) state->wip_block,
                            state->wip_block_capacity + sizeof(struct timeval));
                state->wip_block_len = 0;
            }
        }
    }
}

static void *sdr_routine(void * args) {
    rtlsdr_reader_t* state = (rtlsdr_reader_t*) args;

    int r = rtlsdr_read_async(
            state->sdr, sdr_callback, (void *)state,
            RTLSDR_BUF_NUM, RTLSDR_BUF_LENGTH);

    if (state->sdr_running) {
        // Premature exit -- an error occurred
        circbuf_cancel(state->circbuf);
    } else {
        r = 0;
    }

    state->return_code = r;
    return NULL;
}

void rtlsdr_reader_free(rtlsdr_reader_t* state) {
    if (state->sdr != NULL) {
        rtlsdr_close(state->sdr);
    }
    if (state->circbuf != NULL) {
        circbuf_free(state->circbuf);
    }
    if (state->wip_block != NULL) {
        free(state->wip_block);
    }
    if (state->reader_block != NULL) {
        free(state->reader_block);
    }
    free(state);
}

int rtlsdr_reader_next(rtlsdr_reader_t* state) {
    if (state->standby) {
        // Nothing to read
        return 0;
    }

    // copy history
    block_t* output = state->settings.output;
    size_t history_size = state->settings.history_size;
    size_t new_len = state->settings.block_size - history_size;
    // assert(new_len == state->wip_block_capacity)

    memcpy(output->raw_samples,
           output->raw_samples + new_len,
           history_size * 2);

    // retrieve data and metadata in one fetch from RTL-SDR buffer
    // (FIXME: this temporary buffer can be eliminated
    //  with a block-level circular buffer)

    bool success = circbuf_get(state->circbuf,
                               (char *)(state->reader_block),
                               state->wip_block_capacity
                               + sizeof(struct timeval));

    memcpy(output->raw_samples + history_size,
           state->reader_block,
           new_len * 2);

    // Get metadata
    memcpy(&output->timestamp,
           state->reader_block + state->wip_block_capacity,
           sizeof(struct timeval));
    output->index++;

    if (!success) {
        return (state->cancelled ? 1 : -1);
    }

    return 0;
}

int rtlsdr_reader_start(rtlsdr_reader_t* state) {
    // reset state
    circbuf_reset(state->circbuf);
    state->return_code = 0;
    state->cancelled = false;
    state->wip_block_len = 0;

    // start SDR

    // Create RTL-SDR thread
    int r = pthread_create(&state->sdr_thread, NULL, sdr_routine, state);
    if (r != 0) {
        return -1;
    }

    state->sdr_running = true;
    state->return_code = 0;

    return 0;
}

int rtlsdr_reader_stop(rtlsdr_reader_t* state) {
    state->sdr_running = false;
    circbuf_cancel(state->circbuf); // deadlock
    rtlsdr_cancel_async(state->sdr);

    // wait for thread to finish
    pthread_join(state->sdr_thread, NULL);
    return state->return_code;
}

void rtlsdr_reader_cancel(rtlsdr_reader_t* state) {
    state->cancelled = true;
    circbuf_cancel(state->circbuf);
}

reader_t * rtlsdr_reader_new(reader_settings_t reader_settings,
                             rtlsdr_settings_t *rtl_settings) {
    reader_t* reader = malloc(sizeof(reader_t));
    rtlsdr_reader_t* state = malloc(sizeof(rtlsdr_reader_t));
    if (reader == NULL || state == NULL) {
        free(reader);
        free(state);
        return NULL;
    }

    reader->context = state;
    reader->next = (reader_func_t)&rtlsdr_reader_next;
    reader->start = (reader_func_t)&rtlsdr_reader_start;
    reader->stop = (reader_func_t)&rtlsdr_reader_stop;
    reader->cancel = (reader_func_void_t)&rtlsdr_reader_cancel;
    reader->free = (reader_func_void_t)&rtlsdr_reader_free;

    state->settings = reader_settings;
    state->sdr = NULL;
    state->circbuf = NULL;
    state->sdr_running = false;
    state->return_code = 0;
    state->cancelled = false;
    state->standby = false;
    state->wip_block = NULL;
    state->wip_block_len = 0;
    state->wip_block_capacity = ((state->settings.block_size -
                                  state->settings.history_size)
                                 * 2);
    state->reader_block = NULL;

    // reserve extra space for timestamp metadata
    state->wip_block = malloc(state->wip_block_capacity
                              + sizeof(struct timeval));
    if (state->wip_block == NULL) {
        goto fail;
    }

    state->reader_block = malloc(state->wip_block_capacity
                                 + sizeof(struct timeval));
    if (state->reader_block == NULL) {
        goto fail;
    }

    uint32_t device_count = rtlsdr_get_device_count();
    if (device_count == 0) {
        fprintf(stderr, "No supported RTL-SDR devices found.\n");
        goto fail;
    }
    if (rtl_settings->dev_index >= device_count) {
        fprintf(stderr, "RTL-SDR #%d not found\n", device_count);
        goto fail;
    }

    int r = rtlsdr_open(&state->sdr, (uint32_t)rtl_settings->dev_index);
    if (r < 0) {
		fprintf(stderr, "Failed to open RTL-SDR device #%d.\n",
                rtl_settings->dev_index);
        state->sdr = NULL;
        goto fail;
    }

    // set sample rate
    r = rtlsdr_set_sample_rate(state->sdr, rtl_settings->sample_rate);
    if (r < 0) {
		fprintf(stderr, "Failed to set sample rate.\n");
        goto fail;
    }

    // set center frequency
	r = rtlsdr_set_center_freq(state->sdr, rtl_settings->frequency);
    if (r < 0) {
		fprintf(stderr, "Failed to set center frequency.\n");
        goto fail;
    }

    // manual gain mode
    r = rtlsdr_set_tuner_gain_mode(state->sdr, 1);
    if (r < 0) {
        fprintf(stderr, "Failed to enable manual gain.\n");
        goto fail;
    }

    // set gain
    rtl_settings->gain = nearest_gain(state->sdr, rtl_settings->gain);
    r = rtlsdr_set_tuner_gain(state->sdr, rtl_settings->gain);
	if (r != 0) {
		fprintf(stderr, "Failed to set tuner gain.\n");
        goto fail;
    }

    // reset_buffer
	r = rtlsdr_reset_buffer(state->sdr);
	if (r < 0) {
		fprintf(stderr, "WARNING: Failed to reset buffers.\n");
    }

    state->circbuf = circbuf_new(CIRCBUF_SIZE);
    if (state->circbuf == NULL) {
        fprintf(stderr, "Failed to create circular buffer\n");
        return false;
    }

    return reader;

fail:
    rtlsdr_reader_free(state);
    free(reader);
    return NULL;
}

void rtlsdr_reader_print_histogram(reader_t* reader, FILE* output) {
    rtlsdr_reader_t *state = (rtlsdr_reader_t*) reader->context;
    
    unsigned* hist = circbuf_histogram(state->circbuf);
    unsigned overflows = circbuf_overflows(state->circbuf);
    unsigned sum = 0;
    for (int i = 0; i < CIRCBUF_HISTOGRAM_LEN; ++i) sum += hist[i];
    fprintf(output, "Histogram (%%):");
    for (int i = 0; i < CIRCBUF_HISTOGRAM_LEN; ++i) {
        fprintf(output, " %.0f", hist[i] * 100.0 / sum);
    }
    fprintf(output, "\n");
    if (overflows > 0) {
        fprintf(output, "Number of buffer overflows: %u\n", overflows);
    }
}

#ifdef LIBRTLSDR_BIAS_TEE_SUPPORT
int rtlsdr_reader_set_bias_tee(reader_t* reader, bool on) {
    rtlsdr_reader_t *state = (rtlsdr_reader_t*) reader->context;
    return rtlsdr_set_bias_tee(state->sdr, on);
}
#endif /* LIBRTLSDR_BIAS_TEE_SUPPORT */

void rtlsdr_reader_set_standby(reader_t* reader, bool on) {
    rtlsdr_reader_t *state = (rtlsdr_reader_t*) reader->context;
    state->standby = on;
    if (on) {
        circbuf_clear(state->circbuf);
    }
}
