#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bitmap_file.h"

// The beginning bitmap-file header.
typedef struct {
  uint8_t signature[2];
  uint32_t file_size;
  uint16_t reserved_1;
  uint16_t reserved_2;
  uint32_t pixel_array_offset;
} __attribute__((packed)) BitmapHeader;

// A BITMAPINFOHEADER header.
typedef struct {
  uint32_t header_size;
  int32_t image_width;
  int32_t image_height;
  uint16_t planes;
  uint16_t bits_per_pixel;
  uint32_t compression;
  uint32_t bitmap_size;
  uint32_t horizontal_resolution;
  uint32_t vertical_resolution;
  uint32_t color_count;
  uint32_t important_color_count;
} __attribute__((packed)) BitmapInfoHeader;

int SaveBitmapFile(const char *filename, uint8_t *pixel_data, uint32_t w,
    uint32_t h) {
  BitmapHeader header;
  BitmapInfoHeader info_header;
  uint32_t i;
  uint8_t *pixel_row = NULL;
  uint32_t row_size;
  FILE *output = fopen(filename, "wb");
  if (!output) {
    printf("Failed opening output file %s.\n", filename);
    return 0;
  }
  // Rows must be padded to 4-byte amounts.
  row_size = w * 3;
  while ((row_size % 4) != 0) row_size++;
  pixel_row = (uint8_t *) malloc(row_size);
  if (!pixel_row) {
    printf("Failed allocating memory for pixel row.\n");
    fclose(output);
    return 0;
  }
  memset(pixel_row, 0, row_size);
  memset(&header, 0, sizeof(header));
  memset(&info_header, 0, sizeof(info_header));
  header.signature[0] = 'B';
  header.signature[1] = 'M';
  header.file_size = sizeof(header) + sizeof(info_header) + (row_size * h);
  header.pixel_array_offset = sizeof(header) + sizeof(info_header);
  info_header.header_size = sizeof(info_header);
  info_header.image_width = w;
  info_header.image_height = h;
  info_header.planes = 1;
  info_header.bits_per_pixel = 24;
  info_header.compression = 0;
  info_header.bitmap_size = row_size * h;
  info_header.horizontal_resolution = 12480;
  info_header.vertical_resolution = 12480;
  info_header.color_count = 0;
  info_header.important_color_count = 0;
  if (!fwrite(&header, sizeof(header), 1, output)) {
    printf("Failed writing header.\n");
    fclose(output);
    free(pixel_row);
    return 0;
  }
  if (!fwrite(&info_header, sizeof(info_header), 1, output)) {
    printf("Failed writing info header.\n");
    fclose(output);
    free(pixel_row);
    return 0;
  }
  for (i = 0; i < h; i++) {
    // Any padding in the row will remain 0.
    memcpy(pixel_row, pixel_data, w * 3);
    if (!fwrite(pixel_row, row_size, 1, output)) {
      printf("Failed writing image row.\n");
      fclose(output);
      free(pixel_row);
      return 0;
    }
    pixel_data += w * 3;
  }
  fclose(output);
  free(pixel_row);
  return 1;
}
