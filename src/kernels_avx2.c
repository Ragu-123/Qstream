/*
 * QStream - kernels_avx2.c
 * AVX2 + FMA optimized kernels for x86-64.
 * Compiled with -mavx2 -mfma flags.
 */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#ifdef _MSC_VER
  #include <intrin.h>
#else
  #include <immintrin.h>
#endif

#include "qsf/kernels.h"
#include "qsf/quant.h"
#include <math.h>
#include <string.h>

/* ── AVX2 fused dequant-matvec 4-bit ─────────────────────────────── */
void qsf_matvec_4bit_avx2(const void* w, const float* in, float* out,
                           int rows, int cols, int bs) {
    /* For now, delegate to scalar. AVX2 optimization for inner loop
       would unpack 4-bit values into __m256 and use _mm256_fmadd_ps. */
    qsf_matvec_4bit_scalar(w, in, out, rows, cols, bs);
}

/* ── AVX2 fused dequant-matvec 2-bit ─────────────────────────────── */
void qsf_matvec_2bit_avx2(const void* w, const float* in, float* out,
                           int rows, int cols, int bs) {
    qsf_matvec_2bit_scalar(w, in, out, rows, cols, bs);
}

/* ── AVX2 vector add ─────────────────────────────────────────────── */
void qsf_vec_add_avx2(const float* a, const float* b, float* o, int n) {
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(o + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; i++) o[i] = a[i] + b[i];
}

/* ── AVX2 vector mul ─────────────────────────────────────────────── */
void qsf_vec_mul_avx2(const float* a, const float* b, float* o, int n) {
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(o + i, _mm256_mul_ps(va, vb));
    }
    for (; i < n; i++) o[i] = a[i] * b[i];
}

/* ── AVX2 dot product ────────────────────────────────────────────── */
float qsf_dot_avx2(const float* a, const float* b, int n) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);
        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);
    }
    __m256 s = _mm256_add_ps(sum0, sum1);

    /* Horizontal sum */
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 v = _mm_add_ps(lo, hi);
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    float result = _mm_cvtss_f32(v);

    /* Scalar remainder */
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

/* ── AVX2 SiLU ───────────────────────────────────────────────────── */
void qsf_silu_avx2(float* x, int n) {
    /* SiLU with scalar math but AVX2 memory access */
    qsf_silu_scalar(x, n);
}

/* ── AVX2 GELU ───────────────────────────────────────────────────── */
void qsf_gelu_avx2(float* x, int n) {
    qsf_gelu_scalar(x, n);
}

/* ── AVX2 softmax ────────────────────────────────────────────────── */
void qsf_softmax_avx2(const float* in, float* out, int n) {
    if (n <= 0) return;

    /* Find max with AVX2 */
    __m256 vmax = _mm256_set1_ps(-1e30f);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        vmax = _mm256_max_ps(vmax, v);
    }

    /* Horizontal max */
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 m4 = _mm_max_ps(lo, hi);
    m4 = _mm_max_ps(m4, _mm_shuffle_ps(m4, m4, 0x4E));
    m4 = _mm_max_ps(m4, _mm_shuffle_ps(m4, m4, 0xB1));
    float max_val = _mm_cvtss_f32(m4);

    for (; i < n; i++) {
        if (in[i] > max_val) max_val = in[i];
    }

    /* exp and sum (scalar — AVX2 exp not trivially available) */
    float sum = 0.0f;
    for (i = 0; i < n; i++) {
        out[i] = expf(in[i] - max_val);
        sum += out[i];
    }

    if (sum < 1e-30f) {
        float u = 1.0f / (float)n;
        for (i = 0; i < n; i++) out[i] = u;
        return;
    }

    /* Normalize with AVX2 */
    __m256 inv = _mm256_set1_ps(1.0f / sum);
    for (i = 0; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(out + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(v, inv));
    }
    for (; i < n; i++) out[i] /= sum;
}

/* ── AVX2 RMS norm ───────────────────────────────────────────────── */
void qsf_rms_norm_avx2(const float* in, float* out, const float* w,
                        const float* b, int dim, float eps) {
    /* Sum of squares with AVX2 */
    __m256 ss = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < dim; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        ss = _mm256_fmadd_ps(v, v, ss);
    }
    __m128 hi = _mm256_extractf128_ps(ss, 1);
    __m128 lo = _mm256_castps256_ps128(ss);
    __m128 v4 = _mm_add_ps(lo, hi);
    v4 = _mm_hadd_ps(v4, v4);
    v4 = _mm_hadd_ps(v4, v4);
    float total = _mm_cvtss_f32(v4);
    for (; i < dim; i++) total += in[i] * in[i];

    float rms_inv = 1.0f / sqrtf(total / (float)dim + eps);
    __m256 vrms = _mm256_set1_ps(rms_inv);

    for (i = 0; i + 7 < dim; i += 8) {
        __m256 vi = _mm256_loadu_ps(in + i);
        __m256 vw = _mm256_loadu_ps(w + i);
        __m256 res = _mm256_mul_ps(_mm256_mul_ps(vi, vrms), vw);
        _mm256_storeu_ps(out + i, res);
    }
    for (; i < dim; i++) {
        out[i] = in[i] * rms_inv * w[i];
    }
    (void)b;
}

/* ── AVX2 FP16 → FP32 ───────────────────────────────────────────── */
void qsf_fp16_to_fp32_avx2(const uint16_t* in, float* out, int n) {
    int i = 0;
    /* Use F16C: _mm256_cvtph_ps converts 8 fp16 values at once */
    for (; i + 7 < n; i += 8) {
        __m128i vh = _mm_loadu_si128((const __m128i*)(in + i));
        __m256 vf = _mm256_cvtph_ps(vh);
        _mm256_storeu_ps(out + i, vf);
    }
    for (; i < n; i++) {
        out[i] = qsf_fp16_to_fp32(in[i]);
    }
}

#endif /* x86 */
