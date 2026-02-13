#include "qstream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void) {
  puts("qstream commands:");
  puts("  qstream inspect <file.qsf>");
  puts("  qstream demo-create <file.qsf>");
  puts("  qstream matvec-demo <2|4>");
}

static int cmd_inspect(const char *path) {
  qsf_header_t h;
  int rc = qsf_read_header(path, &h);
  if (rc != 0) {
    fprintf(stderr, "failed to read header: %d\n", rc);
    return 1;
  }
  printf("QSF version=%u layers=%u hidden=%u heads=%u vocab=%u quant=%u\n",
         h.version, h.num_layers, h.hidden_dim, h.num_heads, h.vocab_size,
         h.default_quant);
  qsf_layer_index_entry_t *idx = calloc(h.num_layers, sizeof(*idx));
  if (!idx) {
    return 2;
  }
  rc = qsf_read_layer_index(path, &h, idx, h.num_layers);
  if (rc != 0) {
    fprintf(stderr, "failed to read layer index: %d\n", rc);
    free(idx);
    return 3;
  }
  for (uint32_t i = 0; i < h.num_layers; ++i) {
    printf("layer[%u]: off=%llu comp=%u decomp=%u q=%u c=%u tensors=%u imp=%.3f\n",
           i, (unsigned long long)idx[i].layer_offset, idx[i].compressed_size,
           idx[i].decompressed_size, idx[i].quant_type, idx[i].compression_type,
           idx[i].num_tensors, idx[i].importance);
  }
  free(idx);
  return 0;
}

static int cmd_demo_create(const char *path) {
  qsf_header_t h;
  memset(&h, 0, sizeof(h));
  h.magic = QSF_MAGIC;
  h.version = 1;
  h.header_size = QSF_HEADER_SIZE;
  h.arch = QSF_ARCH_CUSTOM;
  h.num_layers = 2;
  h.hidden_dim = 256;
  h.num_heads = 8;
  h.num_kv_heads = 8;
  h.vocab_size = 32000;
  h.max_seq_len = 1024;
  h.intermediate_dim = 1024;
  h.head_dim = 32;
  h.default_quant = QSF_QUANT_2BIT_ASYM;
  h.activation_type = 1;
  h.norm_type = 1;
  h.position_encoding = 1;
  h.rope_theta = 10000.0f;
  h.layer_index_offset = QSF_HEADER_SIZE;
  h.embedding_offset = QSF_HEADER_SIZE + h.num_layers * sizeof(qsf_layer_index_entry_t);
  h.final_offset = h.embedding_offset + 4096;
  h.bos_token_id = 1;
  h.eos_token_id = 2;
  h.pad_token_id = 0;

  qsf_layer_index_entry_t idx[2];
  memset(idx, 0, sizeof(idx));
  idx[0].layer_offset = h.final_offset + 1024;
  idx[0].compressed_size = 2048;
  idx[0].decompressed_size = 4096;
  idx[0].quant_type = QSF_QUANT_2BIT_ASYM;
  idx[0].compression_type = 1;
  idx[0].num_tensors = 9;
  idx[0].crc32 = 0;
  idx[0].importance = 0.82f;

  idx[1] = idx[0];
  idx[1].layer_offset += 4096;
  idx[1].importance = 0.78f;

  FILE *f = fopen(path, "wb");
  if (!f) {
    return 1;
  }

  fseek(f, 0, SEEK_END);
  long before = ftell(f);
  (void)before;
  h.header_crc32 = 0;
  h.header_crc32 = qs_crc32(&h, 96);

  fwrite(&h, 1, sizeof(h), f);
  fwrite(idx, sizeof(idx[0]), 2, f);

  uint8_t zeros[4096] = {0};
  fwrite(zeros, 1, sizeof(zeros), f);
  fwrite(zeros, 1, sizeof(zeros), f);
  fwrite(zeros, 1, sizeof(zeros), f);
  long end = ftell(f);
  h.total_file_size_low32 = (uint32_t)end;

  rewind(f);
  h.header_crc32 = 0;
  h.header_crc32 = qs_crc32(&h, 96);
  fwrite(&h, 1, sizeof(h), f);
  fclose(f);
  return 0;
}

static int cmd_matvec_demo(int bits) {
  const uint32_t rows = 4;
  const uint32_t cols = 64;
  float input[rows * cols];
  for (uint32_t i = 0; i < rows * cols; ++i) {
    input[i] = (float)(i % 17) / 17.0f;
  }
  float out[rows];
  float scales[4] = {0.1f, 0.2f, 0.1f, 0.15f};
  float mins[4] = {-0.5f, -0.4f, -0.3f, -0.2f};

  if (bits == 4) {
    uint8_t packed[rows * cols / 2u];
    for (size_t i = 0; i < sizeof(packed); ++i) packed[i] = (uint8_t)(i * 13u);
    qs_fused_matvec_4bit_scalar(packed, scales, mins, input, out, rows, cols);
  } else {
    uint8_t packed[rows * cols / 4u];
    for (size_t i = 0; i < sizeof(packed); ++i) packed[i] = (uint8_t)(i * 7u);
    qs_fused_matvec_2bit_scalar(packed, scales, mins, input, out, rows, cols);
  }
  float sum = 0.0f;
  for (uint32_t i = 0; i < rows; ++i) {
    sum += out[i];
  }
  printf("matvec-%dbit checksum=%f\n", bits, sum);
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    usage();
    return 1;
  }

  if (strcmp(argv[1], "inspect") == 0 && argc == 3) {
    return cmd_inspect(argv[2]);
  }
  if (strcmp(argv[1], "demo-create") == 0 && argc == 3) {
    return cmd_demo_create(argv[2]);
  }
  if (strcmp(argv[1], "matvec-demo") == 0 && argc == 3) {
    return cmd_matvec_demo(atoi(argv[2]));
  }

  usage();
  return 1;
}
