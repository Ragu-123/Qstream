/*
 * QStream - format.h
 * QSF file format reader/validator. Parsed model metadata.
 */
#ifndef QSF_FORMAT_H
#define QSF_FORMAT_H

#include "types.h"
#include "error.h"
#include "file_access.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Runtime model descriptor (parsed from QSF file) */
typedef struct {
    QSFHeader        header;
    QSFLayerIndex*   layer_index;     /* [num_layers] */
    FileAccess       file;

    /* Derived values (computed from header) */
    uint32_t head_dim;
    uint32_t num_kv_heads;
    int      is_gated_ffn;
    int      uses_rope;
    int      uses_alibi;
    int      has_bias_attn_qkv;
    int      has_bias_attn_o;
    int      has_bias_ffn;
    int      has_bias_norm;
    int      weight_tied;
} QSFModel;

/* Load and validate a QSF model file. */
QSFError qsf_model_load(QSFModel* model, const char* path, int allow_mmap);

/* Free all resources associated with a loaded model. */
void     qsf_model_free(QSFModel* model);

/* Validate header fields. Returns QSF_OK or error code. */
QSFError qsf_validate_header(const QSFHeader* h);

/* CRC-32 (ISO 3309) */
uint32_t qsf_crc32(const void* data, size_t len);

/*
 * Decompress a layer's data. If uncompressed, returns pointer directly.
 * If LZ4-compressed, decompresses into 'dst' (must be >= decompressed_size).
 * Returns pointer to decompressed data, or NULL on failure.
 */
const void* qsf_decompress_layer(const QSFModel* model,
                                  uint32_t layer_idx,
                                  void* dst, size_t dst_size);

#ifdef __cplusplus
}
#endif
#endif /* QSF_FORMAT_H */
