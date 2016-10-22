// Read raw binary samples from a file

#ifndef RAW_READER_H
#define RAW_READER_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>

#include "reader.h"

reader_t * raw_reader_new(reader_settings_t settings,
                          FILE* file);

#ifdef __cplusplus
}
#endif

#endif /* RAW_READER_H */
