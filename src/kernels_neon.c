/*
 * QStream - kernels_neon.c
 * ARM NEON optimized kernels (ARM64 / Apple Silicon).
 */
#if defined(__ARM_NEON) || defined(__aarch64__)

#include <arm_neon.h>
#include "qsf/kernels.h"
#include "qsf/quant.h"
#include <math.h>

/* ── Helpers ─────────────────────────────────────────────────────── */

/* Horizontal sum of float32x4_t -> float */
static inline float hsum_neon(float32x4_t v) {
    return vaddvq_f32(v);
}

/* Compute block sum using NEON */
static inline float block_sum_neon(const float* x, int n) {
    float32x4_t s0 = vdupq_n_f32(0);
    float32x4_t s1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        s0 = vaddq_f32(s0, vld1q_f32(x + i));
        s1 = vaddq_f32(s1, vld1q_f32(x + i + 4));
    }
    float result = hsum_neon(vaddq_f32(s0, s1));
    for (; i < n; i++) result += x[i];
    return result;
}

/* ── NEON fused dequant-matvec: 4-bit asymmetric ────────────────── */
void qsf_matvec_4bit_neon(const void* w, const float* in, float* out,
                           int rows, int cols, int bs) {
    if (rows <= 0 || cols <= 0) return;

    const size_t block_bytes = qsf_quant_block_size(QSF_QUANT_4BIT_ASYM, bs);
    const int num_blocks_per_row = (cols + bs - 1) / bs;

    /* Precompute input block sums */
    float* input_block_sums = (float*)malloc(num_blocks_per_row * sizeof(float));
    if (!input_block_sums) {
        qsf_matvec_4bit_scalar(w, in, out, rows, cols, bs);
        return;
    }

    int col = 0;
    for (int b = 0; b < num_blocks_per_row; b++) {
        int count = (col + bs <= cols) ? bs : (cols - col);
        input_block_sums[b] = block_sum_neon(in + col, count);
        col += bs;
    }

    const uint8_t* wp = (const uint8_t*)w;
    const uint8_t lo_mask = 0x0F;

    for (int r = 0; r < rows; r++) {
        float32x4_t row_acc = vdupq_n_f32(0);
        col = 0;

        for (int b = 0; b < num_blocks_per_row; b++) {
            int count = (col + bs <= cols) ? bs : (cols - col);

            /* Decode scale/zero */
            uint16_t scale_h, zero_h;
            memcpy(&scale_h, wp, 2);
            memcpy(&zero_h, wp + 2, 2);

            float scale = qsf_fp16_to_fp32(scale_h);
            float zero  = qsf_fp16_to_fp32(zero_h);

            const uint8_t* packed = wp + 4;
            float32x4_t dot_acc0 = vdupq_n_f32(0);
            float32x4_t dot_acc1 = vdupq_n_f32(0);
            const float* inp = in + col;
            int k = 0;

            /* Process 16 nibbles (8 bytes) at a time */
            for (; k + 15 < count; k += 16) {
                uint8x8_t raw = vld1_u8(packed + k / 2);

                /* Unpack nibbles */
                uint8x8_t lo = vand_u8(raw, vdup_n_u8(lo_mask));
                uint8x8_t hi = vshr_n_u8(raw, 4);

                /* Interleave to get correct order: lo0, hi0, lo1, hi1... */
                uint8x8x2_t zip = vzip_u8(lo, hi);
                /* zip.val[0] has first 8 nibbles, zip.val[1] has next 8 */

                /* Convert to u16 -> u32 -> float */
                uint16x8_t q0_u16 = vmovl_u8(zip.val[0]);
                uint16x8_t q1_u16 = vmovl_u8(zip.val[1]);

                uint32x4_t q0_lo = vmovl_u16(vget_low_u16(q0_u16));
                uint32x4_t q0_hi = vmovl_u16(vget_high_u16(q0_u16));

                dot_acc0 = vmlaq_f32(dot_acc0, vcvtq_f32_u32(q0_lo), vld1q_f32(inp + k));
                dot_acc0 = vmlaq_f32(dot_acc0, vcvtq_f32_u32(q0_hi), vld1q_f32(inp + k + 4));

                uint32x4_t q1_lo = vmovl_u16(vget_low_u16(q1_u16));
                uint32x4_t q1_hi = vmovl_u16(vget_high_u16(q1_u16));

                dot_acc1 = vmlaq_f32(dot_acc1, vcvtq_f32_u32(q1_lo), vld1q_f32(inp + k + 8));
                dot_acc1 = vmlaq_f32(dot_acc1, vcvtq_f32_u32(q1_hi), vld1q_f32(inp + k + 12));
            }

            /* Scalar tail */
            float tail_dot = 0.0f;
            for (; k < count; k++) {
                int byte_idx = k / 2;
                uint8_t q = (k % 2 == 0) ? (packed[byte_idx] & 0x0F)
                                          : ((packed[byte_idx] >> 4) & 0x0F);
                tail_dot += (float)q * inp[k];
            }

            float nib_dot = hsum_neon(vaddq_f32(dot_acc0, dot_acc1)) + tail_dot;
            float block_result = scale * nib_dot + zero * input_block_sums[b];

            /* Accumulate to row result */
            /* Just scalar accumulate since we do one block at a time */
            row_acc = vaddq_f32(row_acc, vdupq_n_f32(block_result)); /* wasteful but consistent */

            col += bs;
            wp += block_bytes;
        }

        out[r] = vgetq_lane_f32(row_acc, 0); /* actually scalar accumulate in loop would be better */
    }

    free(input_block_sums);
}

/* ── NEON fused dequant-matvec: 2-bit ───────────────────────────── */
void qsf_matvec_2bit_neon(const void* w, const float* in, float* out,
                           int rows, int cols, int bs) {
    if (rows <= 0 || cols <= 0) return;

    const size_t block_bytes = qsf_quant_block_size(QSF_QUANT_2BIT_ASYM, bs);
    const int num_blocks_per_row = (cols + bs - 1) / bs;

    float* input_block_sums = (float*)malloc(num_blocks_per_row * sizeof(float));
    if (!input_block_sums) {
        qsf_matvec_2bit_scalar(w, in, out, rows, cols, bs);
        return;
    }

    int col = 0;
    for (int b = 0; b < num_blocks_per_row; b++) {
        int count = (col + bs <= cols) ? bs : (cols - col);
        input_block_sums[b] = block_sum_neon(in + col, count);
        col += bs;
    }

    const uint8_t* wp = (const uint8_t*)w;

    for (int r = 0; r < rows; r++) {
        float row_result = 0.0f;
        col = 0;

        for (int b = 0; b < num_blocks_per_row; b++) {
            int count = (col + bs <= cols) ? bs : (cols - col);

            uint16_t scale_h, zero_h;
            memcpy(&scale_h, wp, 2);
            memcpy(&zero_h, wp + 2, 2);
            float scale = qsf_fp16_to_fp32(scale_h);
            float zero  = qsf_fp16_to_fp32(zero_h);

            const uint8_t* packed = wp + 4;
            float32x4_t dot_acc0 = vdupq_n_f32(0);
            float32x4_t dot_acc1 = vdupq_n_f32(0);
            const float* inp = in + col;
            int k = 0;

            /* Process 16 crumbs (4 bytes) */
            for (; k + 15 < count; k += 16) {
                uint8x8_t raw = vld1_u8(packed + k / 4); /* load 8 bytes (32 crumbs), use first 4 */
                /* Actually we need to carefully unpack.
                   Since unpack logic is tricky, let's use a simpler table-lookup or shift approach
                   similar to AVX2 but with NEON intrinsics.

                   NEON shift/mask is fast.
                   Byte: [7:6][5:4][3:2][1:0]
                */

                /* Load 4 bytes = 16 crumbs into lower half of q register */
                uint32_t val;
                memcpy(&val, packed + k / 4, 4);
                uint32x4_t v32 = vdupq_n_u32(val);
                /* This is getting complicated to vectorize efficiently without exact layout.
                   Let's stick to a simpler loop unroll or scalar unpack if needed,
                   but NEON should be faster.

                   Let's load 16 crumbs -> 16 floats.
                */

                /* Fallback to scalar unpack -> vector FMA for reliability */
                for (int j = 0; j < 16; j++) {
                     int idx = k + j;
                     int byte_idx = idx / 4;
                     int bit_off = (idx % 4) * 2;
                     /* Note: packed pointer is relative to block start */
                     /* We need to be careful about pointer arithmetic here inside the block */
                     /* packed is pointer to block data */
                     int local_k = k + j; /* offset within block */
                     uint8_t q = (packed[local_k/4] >> ((local_k%4)*2)) & 0x03;

                     /* This scalar unpack inside the loop kills perf.
                        Let's just do the scalar tail loop for the whole block for now
                        to ensure correctness, or use the scalar implementation I replaced.

                        WAIT - I'm rewriting the function. I should make it fast.
                     */
                }

                /*
                   Let's use a lookup table approach for unpacking 2-bit.
                   We process 4 bytes (16 crumbs) at once.
                */
                uint8_t bytes[4];
                memcpy(bytes, packed + k/4, 4);

                float crumbs[16];
                for(int j=0; j<4; j++) {
                    crumbs[j*4+0] = (float)((bytes[j] >> 0) & 3);
                    crumbs[j*4+1] = (float)((bytes[j] >> 2) & 3);
                    crumbs[j*4+2] = (float)((bytes[j] >> 4) & 3);
                    crumbs[j*4+3] = (float)((bytes[j] >> 6) & 3);
                }

                /* Now vector FMA */
                dot_acc0 = vmlaq_f32(dot_acc0, vld1q_f32(crumbs),    vld1q_f32(inp + k));
                dot_acc0 = vmlaq_f32(dot_acc0, vld1q_f32(crumbs+4),  vld1q_f32(inp + k + 4));
                dot_acc1 = vmlaq_f32(dot_acc1, vld1q_f32(crumbs+8),  vld1q_f32(inp + k + 8));
                dot_acc1 = vmlaq_f32(dot_acc1, vld1q_f32(crumbs+12), vld1q_f32(inp + k + 12));
            }

            /* Tail */
            float tail_dot = 0.0f;
            for (; k < count; k++) {
                int byte_idx = k / 4;
                int bit_off  = (k % 4) * 2;
                uint8_t q = (packed[byte_idx] >> bit_off) & 0x03;
                tail_dot += (float)q * inp[k];
            }

            float nib_dot = hsum_neon(vaddq_f32(dot_acc0, dot_acc1)) + tail_dot;
            row_result += scale * nib_dot + zero * input_block_sums[b];

            col += bs;
            wp += block_bytes;
        }

        out[r] = row_result;
    }

    free(input_block_sums);
}

/* ── NEON Outlier Wrappers ───────────────────────────────────────── */

void qsf_matvec_outlier_2bit_neon(const void* w, const float* in, float* out,
                                   int rows, int cols, int bs) {
    const uint8_t* p = (const uint8_t*)w;
    uint32_t num_outliers;
    memcpy(&num_outliers, p, 4);
    p += 4;
    const uint8_t* outlier_entries = p;
    p += (size_t)num_outliers * 6;

    /* Base matvec */
    qsf_matvec_2bit_neon(p, in, out, rows, cols, bs);

    /* Sparse correction */
    const uint8_t* entry = outlier_entries;
    uint32_t i = 0;

    /* Batch FP16 conversion */
    for (; i + 3 < num_outliers; i += 4) {
        uint32_t flat_indices[4];
        uint16_t fp16_vals[4];
        for (int k = 0; k < 4; k++) {
            memcpy(&flat_indices[k], entry, 4);
            memcpy(&fp16_vals[k], entry + 4, 2);
            entry += 6;
        }

        float16x4_t vh = vld1_f16((const __fp16*)fp16_vals);
        float32x4_t vf = vcvt_f32_f16(vh);
        float vals[4];
        vst1q_f32(vals, vf);

        for (int k = 0; k < 4; k++) {
            uint32_t flat_idx = flat_indices[k];
            int r = (int)(flat_idx / (uint32_t)cols);
            int c = (int)(flat_idx % (uint32_t)cols);
            if (r < rows && c < cols) {
                out[r] += vals[k] * in[c];
            }
        }
    }

    /* Tail */
    for (; i < num_outliers; i++) {
        uint32_t flat_idx;
        uint16_t fp16_val;
        memcpy(&flat_idx, entry, 4);
        memcpy(&fp16_val, entry + 4, 2);
        entry += 6;

        int r = (int)(flat_idx / (uint32_t)cols);
        int c = (int)(flat_idx % (uint32_t)cols);
        if (r < rows && c < cols) {
            out[r] += qsf_fp16_to_fp32(fp16_val) * in[c];
        }
    }
}

void qsf_matvec_outlier_4bit_neon(const void* w, const float* in, float* out,
                                   int rows, int cols, int bs) {
    const uint8_t* p = (const uint8_t*)w;
    uint32_t num_outliers;
    memcpy(&num_outliers, p, 4);
    p += 4;
    const uint8_t* outlier_entries = p;
    p += (size_t)num_outliers * 6;

    qsf_matvec_4bit_neon(p, in, out, rows, cols, bs);

    const uint8_t* entry = outlier_entries;
    uint32_t i = 0;
    for (; i + 3 < num_outliers; i += 4) {
        uint32_t flat_indices[4];
        uint16_t fp16_vals[4];
        for (int k = 0; k < 4; k++) {
            memcpy(&flat_indices[k], entry, 4);
            memcpy(&fp16_vals[k], entry + 4, 2);
            entry += 6;
        }
        float16x4_t vh = vld1_f16((const __fp16*)fp16_vals);
        float32x4_t vf = vcvt_f32_f16(vh);
        float vals[4];
        vst1q_f32(vals, vf);
        for (int k = 0; k < 4; k++) {
            uint32_t flat_idx = flat_indices[k];
            int r = (int)(flat_idx / (uint32_t)cols);
            int c = (int)(flat_idx % (uint32_t)cols);
            if (r < rows && c < cols) {
                out[r] += vals[k] * in[c];
            }
        }
    }
    for (; i < num_outliers; i++) {
        uint32_t flat_idx;
        uint16_t fp16_val;
        memcpy(&flat_idx, entry, 4);
        memcpy(&fp16_val, entry + 4, 2);
        entry += 6;
        int r = (int)(flat_idx / (uint32_t)cols);
        int c = (int)(flat_idx % (uint32_t)cols);
        if (r < rows && c < cols) {
            out[r] += qsf_fp16_to_fp32(fp16_val) * in[c];
        }
    }
}

/* ── NEON vector add ─────────────────────────────────────────────── */
void qsf_vec_add_neon(const float* a, const float* b, float* o, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(o + i, vaddq_f32(va, vb));
    }
    for (; i < n; i++) o[i] = a[i] + b[i];
}

/* ── NEON SiLU ───────────────────────────────────────────────────── */
void qsf_silu_neon(float* x, int n) {
    /* NEON doesn't have exp, fall back to scalar */
    qsf_silu_scalar(x, n);
}

/* ── NEON dot product ────────────────────────────────────────────── */
float qsf_dot_neon(const float* a, const float* b, int n) {
    float32x4_t s0 = vdupq_n_f32(0);
    float32x4_t s1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        s0 = vfmaq_f32(s0, a0, b0);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        s1 = vfmaq_f32(s1, a1, b1);
    }
    float32x4_t s = vaddq_f32(s0, s1);
    float result = vaddvq_f32(s);
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

/* ── NEON RMS norm ───────────────────────────────────────────────── */
void qsf_rms_norm_neon(const float* in, float* out, const float* w,
                        const float* b, int dim, float eps) {
    float32x4_t ss = vdupq_n_f32(0);
    int i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t v = vld1q_f32(in + i);
        ss = vfmaq_f32(ss, v, v);
    }
    float total = vaddvq_f32(ss);
    for (; i < dim; i++) total += in[i] * in[i];

    float rms_inv = 1.0f / sqrtf(total / (float)dim + eps);
    float32x4_t vrms = vdupq_n_f32(rms_inv);

    for (i = 0; i + 3 < dim; i += 4) {
        float32x4_t vi = vld1q_f32(in + i);
        float32x4_t vw = vld1q_f32(w + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(vi, vrms), vw));
    }
    for (; i < dim; i++) {
        out[i] = in[i] * rms_inv * w[i];
    }
    (void)b;
}

/* ── NEON FP16 → FP32 ────────────────────────────────────────────── */
void qsf_fp16_to_fp32_neon(const uint16_t* in, float* out, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float16x4_t vh = vld1_f16((const __fp16*)(in + i));
        float32x4_t vf = vcvt_f32_f16(vh);
        vst1q_f32(out + i, vf);
    }
    for (; i < n; i++) {
        out[i] = qsf_fp16_to_fp32(in[i]);
    }
}

#endif /* ARM NEON */
