/*
 * QStream - kernels_scalar.c
 * Portable scalar (non-SIMD) implementations of all compute kernels.
 */
#include "qsf/kernels.h"
#include "qsf/quant.h"
#include <math.h>
#include <string.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Fused dequant-matvec: 2-bit ─────────────────────────────────── */
void qsf_matvec_2bit_scalar(const void* w, const float* in, float* out,
                             int rows, int cols, int bs) {
    size_t block_bytes = qsf_quant_block_size(QSF_QUANT_2BIT_ASYM, bs);
    int num_blocks_per_row = (cols + bs - 1) / bs;
    const uint8_t* wp = (const uint8_t*)w;
    float dequant_buf[256];  /* max block_size=256 */

    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        int col = 0;
        for (int b = 0; b < num_blocks_per_row; b++) {
            int count = (col + bs <= cols) ? bs : (cols - col);
            qsf_dequant_block_2bit(wp, dequant_buf, count);
            for (int k = 0; k < count; k++) {
                acc += dequant_buf[k] * in[col + k];
            }
            wp += block_bytes;
            col += bs;
        }
        out[r] = acc;
    }
}

/* ── Fused dequant-matvec: 3-bit ─────────────────────────────────── */
void qsf_matvec_3bit_scalar(const void* w, const float* in, float* out,
                             int rows, int cols, int bs) {
    size_t block_bytes = qsf_quant_block_size(QSF_QUANT_3BIT_ASYM, bs);
    int num_blocks_per_row = (cols + bs - 1) / bs;
    const uint8_t* wp = (const uint8_t*)w;
    float dequant_buf[256];

    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        int col = 0;
        for (int b = 0; b < num_blocks_per_row; b++) {
            int count = (col + bs <= cols) ? bs : (cols - col);
            qsf_dequant_block_3bit(wp, dequant_buf, count);
            for (int k = 0; k < count; k++) {
                acc += dequant_buf[k] * in[col + k];
            }
            wp += block_bytes;
            col += bs;
        }
        out[r] = acc;
    }
}

/* ── Fused dequant-matvec: 4-bit asymmetric ──────────────────────── */
void qsf_matvec_4bit_scalar(const void* w, const float* in, float* out,
                             int rows, int cols, int bs) {
    size_t block_bytes = qsf_quant_block_size(QSF_QUANT_4BIT_ASYM, bs);
    int num_blocks_per_row = (cols + bs - 1) / bs;
    const uint8_t* wp = (const uint8_t*)w;
    float dequant_buf[256];

    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        int col = 0;
        for (int b = 0; b < num_blocks_per_row; b++) {
            int count = (col + bs <= cols) ? bs : (cols - col);
            qsf_dequant_block_4bit(wp, dequant_buf, count);
            for (int k = 0; k < count; k++) {
                acc += dequant_buf[k] * in[col + k];
            }
            wp += block_bytes;
            col += bs;
        }
        out[r] = acc;
    }
}

/* ── Fused dequant-matvec: 4-bit symmetric ───────────────────────── */
void qsf_matvec_4bit_sym_scalar(const void* w, const float* in, float* out,
                                 int rows, int cols, int bs) {
    size_t block_bytes = qsf_quant_block_size(QSF_QUANT_4BIT_SYM, bs);
    int num_blocks_per_row = (cols + bs - 1) / bs;
    const uint8_t* wp = (const uint8_t*)w;
    float dequant_buf[256];

    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        int col = 0;
        for (int b = 0; b < num_blocks_per_row; b++) {
            int count = (col + bs <= cols) ? bs : (cols - col);
            qsf_dequant_block_4bit_sym(wp, dequant_buf, count);
            for (int k = 0; k < count; k++) {
                acc += dequant_buf[k] * in[col + k];
            }
            wp += block_bytes;
            col += bs;
        }
        out[r] = acc;
    }
}

/* ── Vector operations ───────────────────────────────────────────── */
void qsf_vec_add_scalar_impl(const float* a, const float* b, float* o, int n) {
    for (int i = 0; i < n; i++) o[i] = a[i] + b[i];
}

void qsf_vec_mul_scalar_impl(const float* a, const float* b, float* o, int n) {
    for (int i = 0; i < n; i++) o[i] = a[i] * b[i];
}

void qsf_vec_scale_scalar_impl(const float* x, float s, float* o, int n) {
    for (int i = 0; i < n; i++) o[i] = x[i] * s;
}

void qsf_vec_add_s_scalar_impl(const float* x, float s, float* o, int n) {
    for (int i = 0; i < n; i++) o[i] = x[i] + s;
}

/* ── Activation functions ────────────────────────────────────────── */
void qsf_silu_scalar(float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

void qsf_gelu_scalar(float* x, int n) {
    const float c = 0.7978845608f;  /* sqrt(2/pi) */
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float inner = c * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

void qsf_relu_scalar(float* x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

/* ── Layer norm ──────────────────────────────────────────────────── */
void qsf_layer_norm_scalar(const float* in, float* out, const float* w,
                            const float* b, int dim, float eps) {
    /* Two-pass: compute mean, then variance */
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += in[i];
    mean /= (float)dim;

    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = in[i] - mean;
        var += d * d;
    }
    var /= (float)dim;

    float inv_std = 1.0f / sqrtf(var + eps);

    for (int i = 0; i < dim; i++) {
        out[i] = (in[i] - mean) * inv_std * w[i];
        if (b) out[i] += b[i];
    }
}

/* ── RMS norm ────────────────────────────────────────────────────── */
void qsf_rms_norm_scalar(const float* in, float* out, const float* w,
                          const float* b, int dim, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += in[i] * in[i];
    float rms = sqrtf(ss / (float)dim + eps);
    float inv_rms = 1.0f / rms;

    for (int i = 0; i < dim; i++) {
        out[i] = in[i] * inv_rms * w[i];
    }
    (void)b;  /* RMS norm has no bias */
}

/* ── Softmax ─────────────────────────────────────────────────────── */
void qsf_softmax_scalar(const float* in, float* out, int n) {
    if (n <= 0) return;

    /* Find max for numerical stability */
    float max_val = in[0];
    for (int i = 1; i < n; i++) {
        if (in[i] > max_val) max_val = in[i];
    }

    /* exp and sum */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = expf(in[i] - max_val);
        sum += out[i];
    }

    /* Normalize */
    if (sum < 1e-30f) {
        /* Edge case: all -inf → uniform */
        float u = 1.0f / (float)n;
        for (int i = 0; i < n; i++) out[i] = u;
        return;
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) out[i] *= inv_sum;
}

/* ── Dot product ─────────────────────────────────────────────────── */
float qsf_dot_scalar(const float* a, const float* b, int n) {
    /* 4-accumulator for precision */
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += a[i]   * b[i];
        s1 += a[i+1] * b[i+1];
        s2 += a[i+2] * b[i+2];
        s3 += a[i+3] * b[i+3];
    }
    float s = s0 + s1 + s2 + s3;
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

/* ── FP16 → FP32 bulk (scalar) ───────────────────────────────────── */
void qsf_fp16_to_fp32_scalar(const uint16_t* in, float* out, int n) {
    qsf_fp16_to_fp32_array(in, out, n);
}

/* ── Utility kernels (§4.3) ──────────────────────────────────────── */

/* SIMD-accelerated min/max scan */
void qsf_find_min_max(const float* QSF_RESTRICT data, int count,
                       float* QSF_RESTRICT min_out, float* QSF_RESTRICT max_out) {
    QSF_ASSERT_SIZE(count);
    if (count <= 0) {
        *min_out = 0.0f;
        *max_out = 0.0f;
        return;
    }
    float lo = data[0], hi = data[0];
    for (int i = 1; i < count; i++) {
        if (data[i] < lo) lo = data[i];
        if (data[i] > hi) hi = data[i];
    }
    *min_out = lo;
    *max_out = hi;
}

/* Multi-accumulator sum for precision */
float qsf_sum_float(const float* QSF_RESTRICT data, int count) {
    QSF_ASSERT_SIZE(count);
    if (count <= 0) return 0.0f;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 3 < count; i += 4) {
        s0 += data[i];
        s1 += data[i + 1];
        s2 += data[i + 2];
        s3 += data[i + 3];
    }
    float s = s0 + s1 + s2 + s3;
    for (; i < count; i++) s += data[i];
    return s;
}

/* Set all floats to a specific value (e.g., -INFINITY for masking) */
void qsf_memset_pattern(float* QSF_RESTRICT dst, float value, int count) {
    QSF_ASSERT_SIZE(count);
    for (int i = 0; i < count; i++) dst[i] = value;
}
