#include "qstream.h"

static inline uint8_t extract_4bit(const uint8_t *packed, uint32_t idx) {
  uint8_t byte = packed[idx / 2u];
  return (idx & 1u) ? (byte >> 4u) : (byte & 0x0Fu);
}

static inline uint8_t extract_2bit(const uint8_t *packed, uint32_t idx) {
  uint8_t byte = packed[idx / 4u];
  uint32_t shift = (idx & 3u) * 2u;
  return (byte >> shift) & 0x03u;
}

void qs_fused_matvec_4bit_scalar(const uint8_t *restrict packed,
                                 const float *restrict scales,
                                 const float *restrict mins,
                                 const float *restrict input,
                                 float *restrict output,
                                 uint32_t rows,
                                 uint32_t cols) {
  const uint32_t block = 64;
  const uint32_t blocks_per_row = cols / block;
  for (uint32_t r = 0; r < rows; ++r) {
    float acc = 0.0f;
    uint32_t row_base = r * cols;
    for (uint32_t b = 0; b < blocks_per_row; ++b) {
      float s = scales[r * blocks_per_row + b];
      float m = mins[r * blocks_per_row + b];
      const uint8_t *row_packed = packed + (row_base / 2u) + b * (block / 2u);
      for (uint32_t i = 0; i < block; ++i) {
        float w = m + s * (float)extract_4bit(row_packed, i);
        acc += w * input[row_base + b * block + i];
      }
    }
    output[r] = acc;
  }
}

void qs_fused_matvec_2bit_scalar(const uint8_t *restrict packed,
                                 const float *restrict scales,
                                 const float *restrict mins,
                                 const float *restrict input,
                                 float *restrict output,
                                 uint32_t rows,
                                 uint32_t cols) {
  const uint32_t block = 64;
  const uint32_t blocks_per_row = cols / block;
  for (uint32_t r = 0; r < rows; ++r) {
    float acc = 0.0f;
    uint32_t row_base = r * cols;
    for (uint32_t b = 0; b < blocks_per_row; ++b) {
      float s = scales[r * blocks_per_row + b];
      float m = mins[r * blocks_per_row + b];
      const uint8_t *row_packed = packed + (row_base / 4u) + b * (block / 4u);
      for (uint32_t i = 0; i < block; ++i) {
        float w = m + s * (float)extract_2bit(row_packed, i);
        acc += w * input[row_base + b * block + i];
      }
    }
    output[r] = acc;
  }
}
