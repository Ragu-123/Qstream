/*
 * QStream - kernels.h
 * Compute kernel dispatch table: function pointers to best ISA variant.
 *
 * CONTRACT (per §4.3 of implementation plan):
 *   - All pointer params use 'restrict': output does NOT alias input or weights.
 *   - Internal allocations are 64-byte aligned (guaranteed by arena).
 *   - size=0 is always valid and a no-op.
 *   - size < SIMD width: vectorized loop skipped, scalar cleanup only.
 *   - All kernel size params are int for hot-path performance.
 *     Assert size < INT_MAX in debug mode.
 */
#ifndef QSF_KERNELS_H
#define QSF_KERNELS_H

#include "platform.h"
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <limits.h>

#ifdef __cplusplus
extern "C" {
#endif

/* restrict portability */
#ifdef _MSC_VER
  #define QSF_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
  #define QSF_RESTRICT __restrict__
#else
  #define QSF_RESTRICT
#endif

/* Debug assertions for kernel contracts */
#ifndef NDEBUG
  #define QSF_ASSERT_SIZE(n) assert((n) >= 0 && (size_t)(n) < (size_t)INT_MAX)
  /* Check that [a, a+size) and [b, b+size) don't overlap */
  #define QSF_ASSERT_NO_OVERLAP(a, b, size) \
      assert((const char*)(a) + (size) <= (const char*)(b) || \
             (const char*)(b) + (size) <= (const char*)(a))
#else
  #define QSF_ASSERT_SIZE(n) ((void)0)
  #define QSF_ASSERT_NO_OVERLAP(a, b, size) ((void)0)
#endif

/*
 * Function pointer types for all compute kernels.
 * All pointer params are 'restrict' — caller must ensure no aliasing.
 */

/* Fused dequant → matrix-vector multiply */
typedef void (*qsf_matvec_fn)(
    const void* QSF_RESTRICT weights_q,   /* quantized weight blocks */
    const float* QSF_RESTRICT input,      /* [cols] */
    float* QSF_RESTRICT output,           /* [rows] */
    int rows, int cols, int block_size
);

/* Element-wise operations */
typedef void (*qsf_vec_op_fn)(const float* QSF_RESTRICT a,
                               const float* QSF_RESTRICT b,
                               float* QSF_RESTRICT out, int n);
typedef void (*qsf_vec_scalar_fn)(const float* QSF_RESTRICT x,
                                   float scalar,
                                   float* QSF_RESTRICT out, int n);
typedef void (*qsf_activation_fn)(float* QSF_RESTRICT x, int n);

/* Normalization */
typedef void (*qsf_norm_fn)(const float* QSF_RESTRICT input,
                             float* QSF_RESTRICT output,
                             const float* QSF_RESTRICT weight,
                             const float* QSF_RESTRICT bias,
                             int dim, float epsilon);

/* Softmax */
typedef void (*qsf_softmax_fn)(const float* QSF_RESTRICT input,
                                float* QSF_RESTRICT output, int n);

/* Dot product */
typedef float (*qsf_dot_fn)(const float* QSF_RESTRICT a,
                             const float* QSF_RESTRICT b, int n);

/* FP16 bulk conversion */
typedef void (*qsf_fp16_cvt_fn)(const uint16_t* QSF_RESTRICT in,
                                 float* QSF_RESTRICT out, int n);

/*
 * Kernel dispatch table — one global instance, initialized at startup.
 */
typedef struct {
    /* Fused dequant-matvec per quant type */
    qsf_matvec_fn matvec_2bit;
    qsf_matvec_fn matvec_3bit;
    qsf_matvec_fn matvec_4bit;
    qsf_matvec_fn matvec_4bit_sym;
    qsf_matvec_fn matvec_outlier_2bit;
    qsf_matvec_fn matvec_outlier_4bit;

    /* Vector ops */
    qsf_vec_op_fn     vec_add;
    qsf_vec_op_fn     vec_mul;
    qsf_vec_scalar_fn vec_scale;
    qsf_vec_scalar_fn vec_add_scalar;

    /* Activations */
    qsf_activation_fn silu;
    qsf_activation_fn gelu;
    qsf_activation_fn relu;

    /* Norms */
    qsf_norm_fn layer_norm;
    qsf_norm_fn rms_norm;

    /* Softmax */
    qsf_softmax_fn softmax;

    /* Dot product */
    qsf_dot_fn dot;

    /* FP16 → FP32 */
    qsf_fp16_cvt_fn fp16_to_fp32;
} QSFKernelTable;

/* Initialize kernel table based on detected platform capabilities */
void qsf_kernels_init(QSFKernelTable* kt, const QSFPlatformInfo* platform);

/* ── Utility kernels (§4.3: find_min_max, sum_float, memset_pattern) ─ */
void  qsf_find_min_max(const float* QSF_RESTRICT data, int count,
                        float* QSF_RESTRICT min_out, float* QSF_RESTRICT max_out);
float qsf_sum_float(const float* QSF_RESTRICT data, int count);
void  qsf_memset_pattern(float* QSF_RESTRICT dst, float value, int count);

/* ── Scalar (portable) implementation declarations ───────────────── */
void qsf_matvec_2bit_scalar(const void* QSF_RESTRICT w,
                             const float* QSF_RESTRICT in,
                             float* QSF_RESTRICT out,
                             int rows, int cols, int bs);
void qsf_matvec_3bit_scalar(const void* QSF_RESTRICT w,
                             const float* QSF_RESTRICT in,
                             float* QSF_RESTRICT out,
                             int rows, int cols, int bs);
void qsf_matvec_4bit_scalar(const void* QSF_RESTRICT w,
                             const float* QSF_RESTRICT in,
                             float* QSF_RESTRICT out,
                             int rows, int cols, int bs);
void qsf_matvec_4bit_sym_scalar(const void* QSF_RESTRICT w,
                                 const float* QSF_RESTRICT in,
                                 float* QSF_RESTRICT out,
                                 int rows, int cols, int bs);
void qsf_vec_add_scalar_impl(const float* QSF_RESTRICT a,
                               const float* QSF_RESTRICT b,
                               float* QSF_RESTRICT o, int n);
void qsf_vec_mul_scalar_impl(const float* QSF_RESTRICT a,
                               const float* QSF_RESTRICT b,
                               float* QSF_RESTRICT o, int n);
void qsf_vec_scale_scalar_impl(const float* QSF_RESTRICT x, float s,
                                 float* QSF_RESTRICT o, int n);
void qsf_vec_add_s_scalar_impl(const float* QSF_RESTRICT x, float s,
                                 float* QSF_RESTRICT o, int n);
void qsf_silu_scalar(float* QSF_RESTRICT x, int n);
void qsf_gelu_scalar(float* QSF_RESTRICT x, int n);
void qsf_relu_scalar(float* QSF_RESTRICT x, int n);
void qsf_layer_norm_scalar(const float* QSF_RESTRICT in,
                            float* QSF_RESTRICT out,
                            const float* QSF_RESTRICT w,
                            const float* QSF_RESTRICT b, int dim, float eps);
void qsf_rms_norm_scalar(const float* QSF_RESTRICT in,
                          float* QSF_RESTRICT out,
                          const float* QSF_RESTRICT w,
                          const float* QSF_RESTRICT b, int dim, float eps);
void qsf_softmax_scalar(const float* QSF_RESTRICT in,
                          float* QSF_RESTRICT out, int n);
float qsf_dot_scalar(const float* QSF_RESTRICT a,
                      const float* QSF_RESTRICT b, int n);
void qsf_fp16_to_fp32_scalar(const uint16_t* QSF_RESTRICT in,
                               float* QSF_RESTRICT out, int n);

/* ── AVX2 implementation declarations (compiled with -mavx2 -mfma) ── */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
void qsf_matvec_2bit_avx2(const void* QSF_RESTRICT w,
                           const float* QSF_RESTRICT in,
                           float* QSF_RESTRICT out,
                           int rows, int cols, int bs);
void qsf_matvec_outlier_2bit_avx2(const void* QSF_RESTRICT w,
                                   const float* QSF_RESTRICT in,
                                   float* QSF_RESTRICT out,
                                   int rows, int cols, int bs);
void qsf_matvec_4bit_avx2(const void* QSF_RESTRICT w,
                           const float* QSF_RESTRICT in,
                           float* QSF_RESTRICT out,
                           int rows, int cols, int bs);
void qsf_matvec_outlier_4bit_avx2(const void* QSF_RESTRICT w,
                                   const float* QSF_RESTRICT in,
                                   float* QSF_RESTRICT out,
                                   int rows, int cols, int bs);
void qsf_matvec_4bit_sym_avx2(const void* QSF_RESTRICT w,
                                const float* QSF_RESTRICT in,
                                float* QSF_RESTRICT out,
                                int rows, int cols, int bs);
void qsf_vec_add_avx2(const float* QSF_RESTRICT a,
                       const float* QSF_RESTRICT b,
                       float* QSF_RESTRICT o, int n);
void qsf_vec_mul_avx2(const float* QSF_RESTRICT a,
                       const float* QSF_RESTRICT b,
                       float* QSF_RESTRICT o, int n);
void qsf_silu_avx2(float* QSF_RESTRICT x, int n);
void qsf_gelu_avx2(float* QSF_RESTRICT x, int n);
float qsf_dot_avx2(const float* QSF_RESTRICT a,
                    const float* QSF_RESTRICT b, int n);
void qsf_softmax_avx2(const float* QSF_RESTRICT in,
                       float* QSF_RESTRICT out, int n);
void qsf_rms_norm_avx2(const float* QSF_RESTRICT in,
                        float* QSF_RESTRICT out,
                        const float* QSF_RESTRICT w,
                        const float* QSF_RESTRICT b, int dim, float eps);
void qsf_fp16_to_fp32_avx2(const uint16_t* QSF_RESTRICT in,
                            float* QSF_RESTRICT out, int n);
#endif

/* ── NEON implementation declarations (ARM64) ──────────────────── */
#if defined(__ARM_NEON) || defined(__aarch64__)
void qsf_matvec_2bit_neon(const void* QSF_RESTRICT w,
                           const float* QSF_RESTRICT in,
                           float* QSF_RESTRICT out,
                           int rows, int cols, int bs);
void qsf_matvec_outlier_2bit_neon(const void* QSF_RESTRICT w,
                                   const float* QSF_RESTRICT in,
                                   float* QSF_RESTRICT out,
                                   int rows, int cols, int bs);
void qsf_matvec_4bit_neon(const void* QSF_RESTRICT w,
                           const float* QSF_RESTRICT in,
                           float* QSF_RESTRICT out,
                           int rows, int cols, int bs);
void qsf_matvec_outlier_4bit_neon(const void* QSF_RESTRICT w,
                                   const float* QSF_RESTRICT in,
                                   float* QSF_RESTRICT out,
                                   int rows, int cols, int bs);
void qsf_vec_add_neon(const float* QSF_RESTRICT a,
                       const float* QSF_RESTRICT b,
                       float* QSF_RESTRICT o, int n);
void qsf_silu_neon(float* QSF_RESTRICT x, int n);
float qsf_dot_neon(const float* QSF_RESTRICT a,
                    const float* QSF_RESTRICT b, int n);
void qsf_rms_norm_neon(const float* QSF_RESTRICT in,
                        float* QSF_RESTRICT out,
                        const float* QSF_RESTRICT w,
                        const float* QSF_RESTRICT b, int dim, float eps);
void qsf_fp16_to_fp32_neon(const uint16_t* QSF_RESTRICT in,
                            float* QSF_RESTRICT out, int n);
#endif

#ifdef __cplusplus
}
#endif
#endif /* QSF_KERNELS_H */
