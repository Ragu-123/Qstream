#ifndef QSTREAM_H
#define QSTREAM_H

#include <stddef.h>
#include <stdint.h>

#define QSF_MAGIC 0x51534631u
#define QSF_HEADER_SIZE 128u

typedef enum {
  QSF_ARCH_GPT2 = 0,
  QSF_ARCH_LLAMA = 1,
  QSF_ARCH_MISTRAL = 2,
  QSF_ARCH_PHI = 3,
  QSF_ARCH_CUSTOM = 4,
} qsf_arch_t;

typedef enum {
  QSF_QUANT_2BIT_SYM = 0,
  QSF_QUANT_2BIT_ASYM = 1,
  QSF_QUANT_3BIT = 2,
  QSF_QUANT_4BIT = 3,
  QSF_QUANT_TERNARY = 4,
  QSF_QUANT_MIXED = 5,
} qsf_quant_t;

#pragma pack(push, 1)
typedef struct {
  uint32_t magic;
  uint32_t version;
  uint32_t header_size;
  uint32_t arch;
  uint32_t num_layers;
  uint32_t hidden_dim;
  uint32_t num_heads;
  uint32_t num_kv_heads;
  uint32_t vocab_size;
  uint32_t max_seq_len;
  uint32_t intermediate_dim;
  uint32_t head_dim;
  uint8_t default_quant;
  uint8_t activation_type;
  uint8_t norm_type;
  uint8_t position_encoding;
  float rope_theta;
  uint64_t layer_index_offset;
  uint64_t embedding_offset;
  uint64_t final_offset;
  uint32_t bos_token_id;
  uint32_t eos_token_id;
  uint32_t pad_token_id;
  uint32_t total_file_size_low32;
  uint32_t header_crc32;
  uint8_t reserved[28];
} qsf_header_t;

typedef struct {
  uint64_t layer_offset;
  uint32_t compressed_size;
  uint32_t decompressed_size;
  uint8_t quant_type;
  uint8_t compression_type;
  uint16_t num_tensors;
  uint32_t crc32;
  float importance;
  uint32_t reserved;
} qsf_layer_index_entry_t;
#pragma pack(pop)

typedef struct {
  uint8_t *base;
  size_t total_size;
  size_t used;
} qs_arena_t;

typedef struct {
  int has_sse42;
  int has_avx2;
  int has_avx512f;
  int has_neon;
} qs_cpu_features_t;

uint32_t qs_crc32(const void *data, size_t len);
int qsf_read_header(const char *path, qsf_header_t *out);
int qsf_validate_header(const qsf_header_t *header);
int qsf_read_layer_index(const char *path, const qsf_header_t *header,
                         qsf_layer_index_entry_t *entries,
                         size_t entry_count);

qs_arena_t *qs_arena_create(size_t size_bytes);
void qs_arena_destroy(qs_arena_t *arena);
void *qs_arena_alloc(qs_arena_t *arena, size_t size, size_t alignment);
void qs_arena_reset(qs_arena_t *arena);

qs_cpu_features_t qs_detect_cpu_features(void);

void qs_fused_matvec_4bit_scalar(const uint8_t *restrict packed,
                                 const float *restrict scales,
                                 const float *restrict mins,
                                 const float *restrict input,
                                 float *restrict output,
                                 uint32_t rows,
                                 uint32_t cols);

void qs_fused_matvec_2bit_scalar(const uint8_t *restrict packed,
                                 const float *restrict scales,
                                 const float *restrict mins,
                                 const float *restrict input,
                                 float *restrict output,
                                 uint32_t rows,
                                 uint32_t cols);

#endif
