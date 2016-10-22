#ifndef CARD_READER_H
#define CARD_READER_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>

#include "reader.h"

reader_t * card_reader_new(reader_settings_t settings,
                           FILE* file);

#ifdef __cplusplus
}
#endif

#endif /* CARD_READER_H */
