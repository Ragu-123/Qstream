#include "qstream.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static size_t elem_offset(const qs_kv_cache_t *c, uint32_t layer, uint32_t head,
                          uint32_t pos) {
  size_t slice = (size_t)c->heads * c->window;
  return ((size_t)layer * slice) + ((size_t)head * c->window) + pos;
}

static size_t vec_offset(const qs_kv_cache_t *c, uint32_t layer, uint32_t head,
                         uint32_t pos) {
  return elem_offset(c, layer, head, pos) * c->head_dim;
}

static void quantize_row(const float *in, uint8_t *out, uint32_t n, uint8_t bits,
                         float *scale_out) {
  float max_abs = 1e-6f;
  for (uint32_t i = 0; i < n; ++i) {
    float a = fabsf(in[i]);
    if (a > max_abs) {
      max_abs = a;
    }
  }
  uint32_t levels = (1u << bits) - 1u;
  float scale = max_abs / (float)(levels / 2u);
  if (scale < 1e-8f) {
    scale = 1e-8f;
  }
  *scale_out = scale;

  memset(out, 0, (n * bits + 7u) / 8u);
  for (uint32_t i = 0; i < n; ++i) {
    int q = (int)lroundf(in[i] / scale) + (int)(levels / 2u);
    if (q < 0) {
      q = 0;
    }
    if ((uint32_t)q > levels) {
      q = (int)levels;
    }
    if (bits == 4) {
      uint32_t b = i / 2u;
      if ((i & 1u) == 0u) {
        out[b] = (uint8_t)(q & 0x0F);
      } else {
        out[b] |= (uint8_t)((q & 0x0F) << 4u);
      }
    } else {
      uint32_t b = i / 4u;
      uint32_t shift = (i & 3u) * 2u;
      out[b] |= (uint8_t)((q & 0x03) << shift);
    }
  }
}

static void dequantize_row(const uint8_t *in, float *out, uint32_t n, uint8_t bits,
                           float scale) {
  uint32_t levels = (1u << bits) - 1u;
  int zp = (int)(levels / 2u);
  for (uint32_t i = 0; i < n; ++i) {
    int q;
    if (bits == 4) {
      q = (i & 1u) ? ((in[i / 2u] >> 4u) & 0x0F) : (in[i / 2u] & 0x0F);
    } else {
      q = (in[i / 4u] >> ((i & 3u) * 2u)) & 0x03;
    }
    out[i] = (float)(q - zp) * scale;
  }
}

int qs_kv_cache_init(qs_kv_cache_t *cache, uint32_t layers, uint32_t heads,
                     uint32_t head_dim, uint32_t window, uint8_t bits) {
  if (!cache || layers == 0 || heads == 0 || head_dim == 0 || window == 0) {
    return -1;
  }
  if (!(bits == 2 || bits == 4)) {
    return -2;
  }
  memset(cache, 0, sizeof(*cache));
  cache->layers = layers;
  cache->heads = heads;
  cache->head_dim = head_dim;
  cache->window = window;
  cache->bits = bits;

  size_t total_vectors = (size_t)layers * heads * window;
  size_t row_bytes = (head_dim * bits + 7u) / 8u;
  size_t total_data = total_vectors * row_bytes;

  cache->k_data = (uint8_t *)calloc(total_data, 1);
  cache->v_data = (uint8_t *)calloc(total_data, 1);
  cache->k_scales = (float *)calloc(total_vectors, sizeof(float));
  cache->v_scales = (float *)calloc(total_vectors, sizeof(float));

  if (!cache->k_data || !cache->v_data || !cache->k_scales || !cache->v_scales) {
    qs_kv_cache_destroy(cache);
    return -3;
  }
  return 0;
}

void qs_kv_cache_destroy(qs_kv_cache_t *cache) {
  if (!cache) {
    return;
  }
  free(cache->k_data);
  free(cache->v_data);
  free(cache->k_scales);
  free(cache->v_scales);
  memset(cache, 0, sizeof(*cache));
}

int qs_kv_store(qs_kv_cache_t *cache, uint32_t layer, uint32_t head,
                uint32_t position, const float *key, const float *value) {
  if (!cache || !key || !value || layer >= cache->layers || head >= cache->heads) {
    return -1;
  }
  uint32_t p = position % cache->window;
  size_t eoff = elem_offset(cache, layer, head, p);
  size_t voff = vec_offset(cache, layer, head, p);
  size_t row_bytes = (cache->head_dim * cache->bits + 7u) / 8u;
  quantize_row(key, cache->k_data + eoff * row_bytes, cache->head_dim, cache->bits,
               &cache->k_scales[eoff]);
  quantize_row(value, cache->v_data + eoff * row_bytes, cache->head_dim, cache->bits,
               &cache->v_scales[eoff]);
  (void)voff;
  return 0;
}

int qs_kv_load_key(const qs_kv_cache_t *cache, uint32_t layer, uint32_t head,
                   uint32_t position, float *out_key) {
  if (!cache || !out_key || layer >= cache->layers || head >= cache->heads) {
    return -1;
  }
  uint32_t p = position % cache->window;
  size_t eoff = elem_offset(cache, layer, head, p);
  size_t row_bytes = (cache->head_dim * cache->bits + 7u) / 8u;
  dequantize_row(cache->k_data + eoff * row_bytes, out_key, cache->head_dim,
                 cache->bits, cache->k_scales[eoff]);
  return 0;
}
