// This file defines a simple library for creating bitmap-format image files.
#ifndef BITMAP_FILE_H
#define BITMAP_FILE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

// Saves a bitmap file with the given filename. Requires the pixel data
// containing h rows of w columns. Each pixel should consist of 3 bytes in
// BGR order. Returns 0 on error.
int SaveBitmapFile(const char *filename, uint8_t *pixel_data, uint32_t w,
    uint32_t h);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // BITMAP_FILE_H
