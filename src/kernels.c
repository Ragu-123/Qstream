#include "qstream.h"

#include <math.h>
#include <stddef.h>

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
  const uint32_t bytes_per_row = cols / 2u;
  for (uint32_t r = 0; r < rows; ++r) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks_per_row; ++b) {
      float s = scales[r * blocks_per_row + b];
      float m = mins[r * blocks_per_row + b];
      const uint8_t *row_packed = packed + r * bytes_per_row + b * (block / 2u);
      for (uint32_t i = 0; i < block; ++i) {
        float w = m + s * (float)extract_4bit(row_packed, i);
        acc += w * input[b * block + i];
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
  const uint32_t bytes_per_row = cols / 4u;
  for (uint32_t r = 0; r < rows; ++r) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks_per_row; ++b) {
      float s = scales[r * blocks_per_row + b];
      float m = mins[r * blocks_per_row + b];
      const uint8_t *row_packed = packed + r * bytes_per_row + b * (block / 4u);
      for (uint32_t i = 0; i < block; ++i) {
        float w = m + s * (float)extract_2bit(row_packed, i);
        acc += w * input[b * block + i];
      }
    }
    output[r] = acc;
  }
}

void qs_vec_add(const float *a, const float *b, float *out, uint32_t size) {
  for (uint32_t i = 0; i < size; ++i) out[i] = a[i] + b[i];
}

void qs_vec_mul(const float *a, const float *b, float *out, uint32_t size) {
  for (uint32_t i = 0; i < size; ++i) out[i] = a[i] * b[i];
}

void qs_vec_scale(float *x, float scale, uint32_t size) {
  for (uint32_t i = 0; i < size; ++i) x[i] *= scale;
}

void qs_rms_norm(const float *input, const float *weight, float *output,
                 uint32_t dim, float epsilon) {
  float mean_sq = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    mean_sq += input[i] * input[i];
  }
  mean_sq /= (float)dim;
  float inv = 1.0f / sqrtf(mean_sq + epsilon);
  for (uint32_t i = 0; i < dim; ++i) {
    output[i] = input[i] * inv * weight[i];
  }
}

void qs_silu_inplace(float *x, uint32_t size) {
  for (uint32_t i = 0; i < size; ++i) {
    float xi = x[i];
    x[i] = xi / (1.0f + expf(-xi));
  }
}

void qs_softmax_temperature(const float *logits, float *probs, uint32_t size,
                            float temperature) {
  if (temperature <= 0.0f) {
    temperature = 1.0f;
  }
  float max_v = logits[0] / temperature;
  for (uint32_t i = 1; i < size; ++i) {
    float v = logits[i] / temperature;
    if (v > max_v) {
      max_v = v;
    }
  }
  float sum = 0.0f;
  for (uint32_t i = 0; i < size; ++i) {
    probs[i] = expf((logits[i] / temperature) - max_v);
    sum += probs[i];
  }
  if (sum == 0.0f) {
    float u = 1.0f / (float)size;
    for (uint32_t i = 0; i < size; ++i) probs[i] = u;
    return;
  }
  for (uint32_t i = 0; i < size; ++i) probs[i] /= sum;
}

int qs_top_k_filter(float *logits, uint32_t size, uint32_t k) {
  if (!logits || k == 0 || k > size) {
    return -1;
  }
  for (uint32_t pass = 0; pass < k; ++pass) {
    uint32_t max_i = pass;
    for (uint32_t j = pass + 1; j < size; ++j) {
      if (logits[j] > logits[max_i]) {
        max_i = j;
      }
    }
    float t = logits[pass];
    logits[pass] = logits[max_i];
    logits[max_i] = t;
  }
  for (uint32_t i = k; i < size; ++i) {
    logits[i] = -1e30f;
  }
  return 0;
}

uint32_t qs_sample_argmax(const float *probs, uint32_t size) {
  uint32_t max_i = 0;
  float max_p = probs[0];
  for (uint32_t i = 1; i < size; ++i) {
    if (probs[i] > max_p) {
      max_p = probs[i];
      max_i = i;
    }
  }
  return max_i;
}
