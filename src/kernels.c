/*
 * QStream - kernels.c
 * Kernel dispatch table initialization: select best ISA variant.
 */
#include "qsf/kernels.h"

void qsf_kernels_init(QSFKernelTable* kt, const QSFPlatformInfo* platform) {
    /* Start with scalar (always available) */
    kt->matvec_2bit    = qsf_matvec_2bit_scalar;
    kt->matvec_3bit    = qsf_matvec_3bit_scalar;
    kt->matvec_4bit    = qsf_matvec_4bit_scalar;
    kt->matvec_4bit_sym = qsf_matvec_4bit_sym_scalar;
    kt->matvec_outlier_2bit = qsf_matvec_outlier_2bit; /* quant.c fallback */
    kt->matvec_outlier_4bit = qsf_matvec_outlier_4bit; /* quant.c fallback */
    kt->vec_add        = qsf_vec_add_scalar_impl;
    kt->vec_mul        = qsf_vec_mul_scalar_impl;
    kt->vec_scale      = qsf_vec_scale_scalar_impl;
    kt->vec_add_scalar = qsf_vec_add_s_scalar_impl;
    kt->silu           = qsf_silu_scalar;
    kt->gelu           = qsf_gelu_scalar;
    kt->relu           = qsf_relu_scalar;
    kt->layer_norm     = qsf_layer_norm_scalar;
    kt->rms_norm       = qsf_rms_norm_scalar;
    kt->softmax        = qsf_softmax_scalar;
    kt->dot            = qsf_dot_scalar;
    kt->fp16_to_fp32   = qsf_fp16_to_fp32_scalar;

    /* Upgrade to AVX2 if available */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if (platform->has_avx2 && platform->has_fma) {
        kt->matvec_2bit  = qsf_matvec_2bit_avx2;
        kt->matvec_outlier_2bit = qsf_matvec_outlier_2bit_avx2;
        kt->matvec_4bit  = qsf_matvec_4bit_avx2;
        kt->matvec_outlier_4bit = qsf_matvec_outlier_4bit_avx2;
        kt->matvec_4bit_sym = qsf_matvec_4bit_sym_avx2;
        kt->vec_add      = qsf_vec_add_avx2;
        kt->vec_mul      = qsf_vec_mul_avx2;
        kt->silu         = qsf_silu_avx2;
        kt->gelu         = qsf_gelu_avx2;
        kt->dot          = qsf_dot_avx2;
        kt->softmax      = qsf_softmax_avx2;
        kt->rms_norm     = qsf_rms_norm_avx2;
        if (platform->has_f16c) {
            kt->fp16_to_fp32 = qsf_fp16_to_fp32_avx2;
        }
    }
#endif

    /* Upgrade to NEON if available */
#if defined(__ARM_NEON) || defined(__aarch64__)
    if (platform->has_neon) {
        kt->matvec_2bit  = qsf_matvec_2bit_neon;
        kt->matvec_outlier_2bit = qsf_matvec_outlier_2bit_neon;
        kt->matvec_4bit  = qsf_matvec_4bit_neon;
        kt->matvec_outlier_4bit = qsf_matvec_outlier_4bit_neon;
        kt->vec_add      = qsf_vec_add_neon;
        kt->silu         = qsf_silu_neon;
        kt->dot          = qsf_dot_neon;
        kt->rms_norm     = qsf_rms_norm_neon;
        kt->fp16_to_fp32 = qsf_fp16_to_fp32_neon;
    }
#endif

    (void)platform;  /* suppress unused on platforms without SIMD */
}
