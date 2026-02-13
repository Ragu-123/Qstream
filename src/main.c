#include "qstream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void) {
  puts("qstream commands:");
  puts("  qstream inspect <file.qsf>");
  puts("  qstream demo-create <file.qsf>");
  puts("  qstream single-phase-demo <file.qsf>");
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
  h.max_seq_len = 512;
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
  for (uint32_t i = 0; i < 2; ++i) {
    idx[i].layer_offset = h.final_offset + 1024u + 4096u * i;
    idx[i].compressed_size = 2048;
    idx[i].decompressed_size = 4096;
    idx[i].quant_type = QSF_QUANT_2BIT_ASYM;
    idx[i].compression_type = 0;
    idx[i].num_tensors = 9;
    idx[i].importance = 0.8f - 0.02f * (float)i;
  }

  FILE *f = fopen(path, "wb");
  if (!f) {
    return 1;
  }

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

static int cmd_single_phase_demo(const char *path) {
  qs_cpu_features_t f = qs_detect_cpu_features();
  printf("cpu features: sse4.2=%d avx2=%d avx512f=%d neon=%d\n", f.has_sse42,
         f.has_avx2, f.has_avx512f, f.has_neon);

  qs_model_t model;
  if (qs_model_open(path, &model) != 0) {
    fprintf(stderr, "model open failed\n");
    return 1;
  }

  qs_layer_stream_t stream;
  if (qs_stream_init(&model, &stream) != 0) {
    qs_model_close(&model);
    return 2;
  }
  for (uint32_t i = 0; i < model.header.num_layers; ++i) {
    if (qs_stream_load_layer(&stream, i) != 0) {
      qs_stream_destroy(&stream);
      qs_model_close(&model);
      return 3;
    }
  }

  qs_kv_cache_t kv;
  if (qs_kv_cache_init(&kv, model.header.num_layers, model.header.num_kv_heads,
                       model.header.head_dim, 64, 2) != 0) {
    qs_stream_destroy(&stream);
    qs_model_close(&model);
    return 4;
  }

  float key[32], value[32], loaded[32];
  for (uint32_t i = 0; i < 32; ++i) {
    key[i] = ((float)(i % 7) - 3.0f) / 3.0f;
    value[i] = ((float)(i % 11) - 5.0f) / 5.0f;
  }
  qs_kv_store(&kv, 0, 0, 0, key, value);
  qs_kv_load_key(&kv, 0, 0, 0, loaded);

  float rms_w[32], rms_out[32];
  for (uint32_t i = 0; i < 32; ++i) rms_w[i] = 1.0f;
  qs_rms_norm(loaded, rms_w, rms_out, 32, 1e-5f);
  qs_silu_inplace(rms_out, 32);

  float logits[16], probs[16];
  for (uint32_t i = 0; i < 16; ++i) logits[i] = rms_out[i];
  qs_top_k_filter(logits, 16, 5);
  qs_softmax_temperature(logits, probs, 16, 0.8f);
  uint32_t token = qs_sample_argmax(probs, 16);

  float checksum = 0.0f;
  for (uint32_t i = 0; i < 32; ++i) checksum += rms_out[i];
  printf("single-phase checksum=%f token=%u loaded_layer=%u bytes=%zu\n", checksum,
         token, stream.loaded_layer, stream.loaded_size);

  qs_kv_cache_destroy(&kv);
  qs_stream_destroy(&stream);
  qs_model_close(&model);
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
  if (strcmp(argv[1], "single-phase-demo") == 0 && argc == 3) {
    return cmd_single_phase_demo(argv[2]);
  }

  usage();
  return 1;
}
