#include "qstream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(qsf_header_t) == 128, "qsf_header_t must be 128 bytes");
_Static_assert(sizeof(qsf_layer_index_entry_t) == 32, "layer index entry must be 32 bytes");

static uint32_t crc32_table[256];
static int crc32_init_done = 0;

static void crc32_init(void) {
  if (crc32_init_done) {
    return;
  }
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int j = 0; j < 8; ++j) {
      c = (c & 1u) ? (0xEDB88320u ^ (c >> 1u)) : (c >> 1u);
    }
    crc32_table[i] = c;
  }
  crc32_init_done = 1;
}

uint32_t qs_crc32(const void *data, size_t len) {
  crc32_init();
  const uint8_t *p = (const uint8_t *)data;
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; ++i) {
    c = crc32_table[(c ^ p[i]) & 0xFFu] ^ (c >> 8u);
  }
  return c ^ 0xFFFFFFFFu;
}

int qsf_validate_header(const qsf_header_t *header) {
  if (!header) {
    return -1;
  }
  if (header->magic != QSF_MAGIC) {
    return -2;
  }
  if (header->version != 1u || header->header_size != QSF_HEADER_SIZE) {
    return -3;
  }
  qsf_header_t temp = *header;
  temp.header_crc32 = 0;
  uint32_t crc = qs_crc32(&temp, 96);
  if (crc != header->header_crc32) {
    return -4;
  }
  if (header->num_layers == 0 || header->hidden_dim == 0) {
    return -5;
  }
  return 0;
}

int qsf_read_header(const char *path, qsf_header_t *out) {
  if (!path || !out) {
    return -1;
  }
  FILE *f = fopen(path, "rb");
  if (!f) {
    return -2;
  }
  size_t n = fread(out, 1, sizeof(*out), f);
  fclose(f);
  if (n != sizeof(*out)) {
    return -3;
  }
  return qsf_validate_header(out);
}

int qsf_read_layer_index(const char *path, const qsf_header_t *header,
                         qsf_layer_index_entry_t *entries,
                         size_t entry_count) {
  if (!path || !header || !entries || entry_count < header->num_layers) {
    return -1;
  }
  FILE *f = fopen(path, "rb");
  if (!f) {
    return -2;
  }
  if (fseek(f, (long)header->layer_index_offset, SEEK_SET) != 0) {
    fclose(f);
    return -3;
  }
  size_t want = (size_t)header->num_layers;
  size_t got = fread(entries, sizeof(qsf_layer_index_entry_t), want, f);
  fclose(f);
  if (got != want) {
    return -4;
  }
  return 0;
}
