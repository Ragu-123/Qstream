/*
 * QStream - format.c
 * QSF file reader, validator, and CRC-32 implementation.
 */
#include "qsf/format.h"
#include "qsf/error.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── CRC-32 (ISO 3309 / ITU-T V.42) ─────────────────────────────── */
static uint32_t crc32_table[256];
static int crc32_table_ready = 0;

static void crc32_init_table(void) {
    for (int i = 0; i < 256; i++) {
        uint32_t crc = (uint32_t)i;
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (crc & 1 ? 0xEDB88320u : 0);
        }
        crc32_table[i] = crc;
    }
    crc32_table_ready = 1;
}

uint32_t qsf_crc32(const void* data, size_t len) {
    if (!crc32_table_ready) crc32_init_table();
    const uint8_t* p = (const uint8_t*)data;
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc = crc32_table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

/* ── Header validation ───────────────────────────────────────────── */
QSFError qsf_validate_header(const QSFHeader* h) {
    if (h->magic != QSF_MAGIC) {
        qsf_set_error(QSF_ERR_FILE_CORRUPTED, "bad magic number");
        return QSF_ERR_FILE_CORRUPTED;
    }
    if (h->version > QSF_VERSION) {
        qsf_set_error(QSF_ERR_FILE_VERSION, "file version too new");
        return QSF_ERR_FILE_VERSION;
    }
    if (h->header_size < QSF_HEADER_SIZE) {
        qsf_set_error(QSF_ERR_INVALID_MODEL, "header_size too small");
        return QSF_ERR_INVALID_MODEL;
    }
    if (h->num_layers == 0 || h->num_layers > QSF_MAX_LAYERS) {
        qsf_set_error(QSF_ERR_INVALID_MODEL, "invalid num_layers");
        return QSF_ERR_INVALID_MODEL;
    }
    if (h->hidden_dim == 0) {
        qsf_set_error(QSF_ERR_INVALID_MODEL, "hidden_dim is 0");
        return QSF_ERR_INVALID_MODEL;
    }
    if (h->num_heads == 0) {
        qsf_set_error(QSF_ERR_INVALID_MODEL, "num_heads is 0");
        return QSF_ERR_INVALID_MODEL;
    }
    if (h->hidden_dim % h->num_heads != 0) {
        qsf_set_error(QSF_ERR_INVALID_MODEL, "hidden_dim not divisible by num_heads");
        return QSF_ERR_INVALID_MODEL;
    }

    uint32_t kv = h->num_kv_heads;
    if (kv == 0) kv = h->num_heads;
    if (h->num_heads % kv != 0) {
        qsf_set_error(QSF_ERR_INVALID_MODEL, "num_heads not divisible by num_kv_heads");
        return QSF_ERR_INVALID_MODEL;
    }

    if (h->vocab_size < 100) {
        qsf_set_error(QSF_ERR_INVALID_MODEL, "vocab_size too small");
        return QSF_ERR_INVALID_MODEL;
    }

    uint32_t bs = h->block_size;
    if (bs == 0) bs = QSF_BLOCK_SIZE;
    if (bs < 32 || bs > 256 || (bs & (bs - 1)) != 0) {
        qsf_set_error(QSF_ERR_INVALID_MODEL, "block_size must be power of 2, 32-256");
        return QSF_ERR_INVALID_MODEL;
    }

    /* Check reserved bytes are zero (warn only) */
    int reserved_nonzero = 0;
    for (int i = 0; i < (int)sizeof(h->reserved); i++) {
        if (h->reserved[i] != 0) { reserved_nonzero = 1; break; }
    }
    if (reserved_nonzero) {
        fprintf(stderr, "[WARN] reserved header bytes are non-zero (newer format?)\n");
    }

    return QSF_OK;
}

/* ── Model loading ───────────────────────────────────────────────── */
QSFError qsf_model_load(QSFModel* model, const char* path, int allow_mmap) {
    memset(model, 0, sizeof(*model));

    /* Open file */
    QSFError err = file_access_open(&model->file, path, allow_mmap);
    if (err != QSF_OK) return err;

    /* Check minimum size */
    if (model->file.file_size < QSF_HEADER_SIZE) {
        qsf_set_error(QSF_ERR_FILE_CORRUPTED, "file too small for header");
        file_access_close(&model->file);
        return QSF_ERR_FILE_CORRUPTED;
    }

    /* Read header */
    const void* hdr_data = file_access_get(&model->file, 0, QSF_HEADER_SIZE);
    if (!hdr_data) {
        qsf_set_error(QSF_ERR_IO_FAILURE, "failed to read header");
        file_access_close(&model->file);
        return QSF_ERR_IO_FAILURE;
    }
    memcpy(&model->header, hdr_data, QSF_HEADER_SIZE);

    /* Validate header */
    err = qsf_validate_header(&model->header);
    if (err != QSF_OK) {
        file_access_close(&model->file);
        return err;
    }

    /* Verify header CRC (first 168 bytes) */
    uint32_t computed_crc = qsf_crc32(&model->header, 168);
    if (computed_crc != model->header.header_crc32) {
        fprintf(stderr, "[WARN] header CRC mismatch (got 0x%08X, expected 0x%08X)\n",
                computed_crc, model->header.header_crc32);
        /* Non-fatal for now — converter might not set CRC yet */
    }

    /* Check file size */
    if (model->header.total_file_size != 0 &&
        model->header.total_file_size != model->file.file_size) {
        fprintf(stderr, "[WARN] file size mismatch: header says %llu, actual %zu\n",
                (unsigned long long)model->header.total_file_size,
                model->file.file_size);
    }

    /* Compute derived values */
    model->head_dim = model->header.head_dim;
    if (model->head_dim == 0) {
        model->head_dim = model->header.hidden_dim / model->header.num_heads;
    }
    model->num_kv_heads = model->header.num_kv_heads;
    if (model->num_kv_heads == 0) {
        model->num_kv_heads = model->header.num_heads;
    }
    model->is_gated_ffn   = (model->header.ffn_type == QSF_FFN_GATED);
    model->uses_rope      = (model->header.pos_enc_type == QSF_POS_ROPE);
    model->uses_alibi     = (model->header.pos_enc_type == QSF_POS_ALIBI);
    model->has_bias_attn_qkv = !!(model->header.has_bias & 0x01);
    model->has_bias_attn_o   = !!(model->header.has_bias & 0x02);
    model->has_bias_ffn      = !!(model->header.has_bias & 0x04);
    model->has_bias_norm     = !!(model->header.has_bias & 0x08);
    model->weight_tied       = (model->header.weight_tying == 1 ||
                                model->header.tie_word_embeddings == 1);

    /* Read layer index */
    uint64_t li_offset = model->header.layer_index_offset;
    size_t   li_size   = (size_t)model->header.num_layers * sizeof(QSFLayerIndex);

    if (li_offset == 0 || li_offset + li_size > model->file.file_size) {
        qsf_set_error(QSF_ERR_FILE_CORRUPTED, "invalid layer index offset");
        file_access_close(&model->file);
        return QSF_ERR_FILE_CORRUPTED;
    }

    model->layer_index = (QSFLayerIndex*)malloc(li_size);
    if (!model->layer_index) {
        qsf_set_error(QSF_ERR_OUT_OF_MEMORY, "layer index alloc");
        file_access_close(&model->file);
        return QSF_ERR_OUT_OF_MEMORY;
    }

    const void* li_data = file_access_get(&model->file, (size_t)li_offset, li_size);
    if (!li_data) {
        qsf_set_error(QSF_ERR_IO_FAILURE, "failed to read layer index");
        free(model->layer_index);
        file_access_close(&model->file);
        return QSF_ERR_IO_FAILURE;
    }
    memcpy(model->layer_index, li_data, li_size);

    /* Validate layer entries */
    for (uint32_t i = 0; i < model->header.num_layers; i++) {
        QSFLayerIndex* li = &model->layer_index[i];
        if (li->compressed_size > li->decompressed_size && li->compression_type != 0) {
            fprintf(stderr, "[WARN] layer %u: compressed > decompressed\n", i);
        }
        if (li->offset + li->compressed_size > model->file.file_size) {
            fprintf(stderr, "[WARN] layer %u: data extends past EOF\n", i);
        }
    }

    return QSF_OK;
}

void qsf_model_free(QSFModel* model) {
    if (!model) return;
    free(model->layer_index);
    model->layer_index = NULL;
    file_access_close(&model->file);
    memset(model, 0, sizeof(*model));
}

/* ── Layer decompression ─────────────────────────────────────────── */
#ifdef QSF_HAS_LZ4
  #include <lz4.h>
#endif

const void* qsf_decompress_layer(const QSFModel* model,
                                  uint32_t layer_idx,
                                  void* dst, size_t dst_size) {
    if (!model || !model->layer_index) return NULL;
    if (layer_idx >= model->header.num_layers) return NULL;

    const QSFLayerIndex* li = &model->layer_index[layer_idx];
    size_t comp_size   = (size_t)li->compressed_size;
    size_t decomp_size = (size_t)li->decompressed_size;

    /* Read compressed/raw data from file */
    const void* src = file_access_get(
        (FileAccess*)&model->file, (size_t)li->offset, comp_size);
    if (!src) {
        qsf_set_error(QSF_ERR_IO_FAILURE, "failed to read layer data");
        return NULL;
    }

    if (li->compression_type == QSF_COMPRESS_NONE) {
        /* Uncompressed: verify CRC and return directly */
        if (li->crc32_compressed != 0) {
            uint32_t actual = qsf_crc32(src, comp_size);
            if (actual != li->crc32_compressed) {
                fprintf(stderr, "[WARN] layer %u CRC mismatch\n", layer_idx);
            }
        }
        return src;
    }

    if (li->compression_type == QSF_COMPRESS_LZ4 ||
        li->compression_type == QSF_COMPRESS_LZ4HC) {
#ifdef QSF_HAS_LZ4
        if (!dst || dst_size < decomp_size) {
            qsf_set_error(QSF_ERR_OUT_OF_MEMORY,
                          "decompress buffer too small");
            return NULL;
        }

        int result = LZ4_decompress_safe(
            (const char*)src, (char*)dst,
            (int)comp_size, (int)decomp_size);

        if (result < 0 || (size_t)result != decomp_size) {
            qsf_set_error(QSF_ERR_FILE_CORRUPTED,
                          "LZ4 decompression failed");
            return NULL;
        }

        /* Verify decompressed CRC */
        if (li->crc32_decompressed != 0) {
            uint32_t actual = qsf_crc32(dst, decomp_size);
            if (actual != li->crc32_decompressed) {
                fprintf(stderr, "[WARN] layer %u decompressed CRC mismatch\n",
                        layer_idx);
            }
        }

        return dst;
#else
        qsf_set_error(QSF_ERR_INTERNAL,
                      "LZ4 compression not supported (compile with QSF_HAS_LZ4)");
        return NULL;
#endif
    }

    qsf_set_error(QSF_ERR_INTERNAL, "unsupported compression type");
    return NULL;
}

