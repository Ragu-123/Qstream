/*
 * QStream - kernels_neon.c
 * ARM NEON optimized kernels (ARM64 / Apple Silicon).
 */
#if defined(__ARM_NEON) || defined(__aarch64__)

#include <arm_neon.h>
#include "qsf/kernels.h"
#include "qsf/quant.h"
#include <math.h>

/* ── NEON fused dequant-matvec (delegates for now) ───────────────── */
void qsf_matvec_2bit_neon(const void* w, const float* in, float* out,
                           int rows, int cols, int bs) {
    qsf_matvec_2bit_scalar(w, in, out, rows, cols, bs);
}

void qsf_matvec_4bit_neon(const void* w, const float* in, float* out,
                           int rows, int cols, int bs) {
    qsf_matvec_4bit_scalar(w, in, out, rows, cols, bs);
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
