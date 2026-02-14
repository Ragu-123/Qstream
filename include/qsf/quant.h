/*
 * QStream - quant.h
 * Quantization / dequantization: block-level pack & unpack.
 */
#ifndef QSF_QUANT_H
#define QSF_QUANT_H

#include "types.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 2-bit block (block_size=64):
 *   scale  (fp16, 2 bytes)
 *   zero   (fp16, 2 bytes)
 *   packed (64 values × 2 bits = 16 bytes)
 *   Total: 20 bytes per block of 64 values
 *
 * 4-bit block:
 *   scale  (fp16, 2 bytes)
 *   zero   (fp16, 2 bytes)
 *   packed (64 values × 4 bits = 32 bytes)
 *   Total: 36 bytes per block
 *
 * 3-bit block:
 *   scale  (fp16, 2 bytes)
 *   zero   (fp16, 2 bytes)
 *   packed (64 values × 3 bits = 24 bytes)
 *   Total: 28 bytes per block
 */

/* Size of a quantization block in bytes (packed meta + data) */
size_t qsf_quant_block_size(QSFQuantType type, int block_size);

/* Dequantize one block into float output.
   block_data points to the packed block (scale + zero + values).
   count: number of valid elements (≤ block_size, for partial blocks).*/
void qsf_dequant_block_2bit(const void* block_data, float* out, int count);
void qsf_dequant_block_3bit(const void* block_data, float* out, int count);
void qsf_dequant_block_4bit(const void* block_data, float* out, int count);
void qsf_dequant_block_4bit_sym(const void* block_data, float* out, int count);

/* Quantize one block of floats.
   values: input floats (block_size elements).
   out: output packed block. */
void qsf_quant_block_2bit(const float* values, void* out, int count);
void qsf_quant_block_3bit(const float* values, void* out, int count);
void qsf_quant_block_4bit(const float* values, void* out, int count);

/*
 * Outlier-aware dequantization.
 * Tensor format: [num_outliers(u32)] [outlier_entries(6*N)] [quantized_blocks]
 * Each outlier entry: [flat_index(u32) + fp16_value(u16)] = 6 bytes
 *
 * Dequantizes quantized blocks, then patches in outlier FP16 values at
 * their original positions. This allows the bulk quantization to use a
 * tighter range (outliers removed), giving much better accuracy.
 */
void qsf_dequant_outlier_2bit(const void* data, size_t data_size,
                                float* out, int total_elements, int block_size);
void qsf_dequant_outlier_4bit(const void* data, size_t data_size,
                                float* out, int total_elements, int block_size);

/*
 * Outlier-aware fused dequant → matvec.
 * Combines outlier patching with matrix-vector multiply in one pass.
 */
void qsf_matvec_outlier_2bit(const void* data, size_t data_size,
                               const float* input, float* output,
                               int rows, int cols, int block_size);
void qsf_matvec_outlier_4bit(const void* data, size_t data_size,
                               const float* input, float* output,
                               int rows, int cols, int block_size);

/* FP16 ↔ FP32 conversions */
float    qsf_fp16_to_fp32(uint16_t h);
uint16_t qsf_fp32_to_fp16(float f);

/* Bulk FP16 → FP32 */
void qsf_fp16_to_fp32_array(const uint16_t* in, float* out, int count);

#ifdef __cplusplus
}
#endif
#endif /* QSF_QUANT_H */
