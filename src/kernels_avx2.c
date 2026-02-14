/*
 * QStream - kernels_avx2.c
 * AVX2 + FMA + F16C optimized compute kernels for x86-64.
 *
 * Compiled with -mavx2 -mfma -mf16c flags.
 *
 * Key optimizations in fused dequant-matvec:
 *   1. Zero-point factoring:
 *      out[r] = Σ_b [ scale_b · dot(nibbles_b, input_b) + zero_b · sum(input_b) ]
 *      The input block sums are precomputed ONCE and amortized across all rows.
 *   2. 16-nibble-at-a-time unpack via low/high split + interleave.
 *   3. 4-accumulator FMA pipeline to saturate port throughput.
 *   4. F16C hardware FP16→FP32 for scale/zero metadata.
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
#include <stdlib.h>

/* ── Helpers ─────────────────────────────────────────────────────── */

/* Horizontal sum of a 256-bit float vector → scalar float */
static inline float hsum_avx(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s  = _mm_add_ps(lo, hi);        /* 4 floats */
    s = _mm_hadd_ps(s, s);                 /* 2 floats */
    s = _mm_hadd_ps(s, s);                 /* 1 float  */
    return _mm_cvtss_f32(s);
}

/* Compute the sum of a float vector using AVX2 */
static inline float block_sum_avx(const float* x, int n) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        s0 = _mm256_add_ps(s0, _mm256_loadu_ps(x + i));
        s1 = _mm256_add_ps(s1, _mm256_loadu_ps(x + i + 8));
    }
    __m256 s = _mm256_add_ps(s0, s1);
    float result = hsum_avx(s);
    for (; i < n; i++) result += x[i];
    return result;
}

/* ── AVX2 fused dequant-matvec: 4-bit asymmetric ────────────────── *
 *
 * Block layout (from quant.c):
 *   [FP16 scale (2B)] [FP16 zero (2B)] [packed nibbles (bs/2 B)]
 *   Even index i → low  nibble: byte[i/2] & 0x0F
 *   Odd  index i → high nibble: byte[i/2] >> 4
 *   Dequant: value = scale * nibble + zero
 *
 * Algebraic factoring (key insight — eliminates per-element zero mul):
 *   out[r] = Σ_blocks [ scale_b · Σ_k(nib_k · in[col+k])
 *                      + zero_b  · Σ_k(in[col+k])          ]
 *
 * Phase 1: Precompute input_block_sums[b] = Σ_k(in[b*bs..b*bs+bs-1])
 *          Done once; cost amortized over all 'rows' output elements.
 *
 * Phase 2: For each row, iterate over blocks:
 *   - F16C decode scale/zero from 2×FP16
 *   - Unpack 16 nibbles at a time via low/high split + interleave
 *   - 4-accumulator FMA: acc += nibble_float * input
 *   - Reduce: out[r] = scale * hsum(acc) + zero * input_block_sum
 */
void qsf_matvec_4bit_avx2(const void* w, const float* in, float* out,
                           int rows, int cols, int bs) {
    if (rows <= 0 || cols <= 0) return;

    const size_t block_bytes = qsf_quant_block_size(QSF_QUANT_4BIT_ASYM, bs);
    const int num_blocks_per_row = (cols + bs - 1) / bs;
    const __m128i lo_mask_128 = _mm_set1_epi8(0x0F);

    /* ── Phase 1: Precompute input block sums ────────────────────── */
    float* input_block_sums = (float*)malloc(num_blocks_per_row * sizeof(float));
    if (!input_block_sums) {
        /* OOM fallback to scalar */
        qsf_matvec_4bit_scalar(w, in, out, rows, cols, bs);
        return;
    }
    {
        int col = 0;
        for (int b = 0; b < num_blocks_per_row; b++) {
            int count = (col + bs <= cols) ? bs : (cols - col);
            input_block_sums[b] = block_sum_avx(in + col, count);
            col += bs;
        }
    }

    /* ── Phase 2: Row-major fused dequant-dot ────────────────────── */
    const uint8_t* wp = (const uint8_t*)w;

    for (int r = 0; r < rows; r++) {
        __m256 row_acc = _mm256_setzero_ps();  /* accumulated: Σ scale*dot + zero*sum */
        int col = 0;

        for (int b = 0; b < num_blocks_per_row; b++) {
            int count = (col + bs <= cols) ? bs : (cols - col);

            /* ── Decode FP16 scale & zero via F16C ─────────────── */
            uint16_t scale_h, zero_h;
            memcpy(&scale_h, wp, 2);
            memcpy(&zero_h, wp + 2, 2);

            /* F16C: convert 2 FP16 values packed in a __m128i */
            __m128i hpair = _mm_set_epi16(0,0,0,0, 0,0, zero_h, scale_h);
            __m128 fpair  = _mm_cvtph_ps(hpair);  /* [scale, zero, 0, 0] */
            float scale = _mm_cvtss_f32(fpair);
            float zero  = _mm_cvtss_f32(_mm_shuffle_ps(fpair, fpair, 0x55));

            const uint8_t* packed = wp + 4;

            /* ── Fused nibble-dot: Σ nibble_k * input[col+k] ───── */
            __m256 dot_acc0 = _mm256_setzero_ps();
            __m256 dot_acc1 = _mm256_setzero_ps();
            __m256 dot_acc2 = _mm256_setzero_ps();
            __m256 dot_acc3 = _mm256_setzero_ps();

            const float* inp = in + col;
            int k = 0;

            /*
             * Process 32 nibbles (16 bytes packed) at a time → 4 × __m256 FMA.
             * 16 packed bytes → 32 nibbles → 32 floats → 4 groups of 8.
             *
             * Nibble unpack strategy (per 8 packed bytes → 16 nibbles → 2×__m256):
             *   1. Load 8 bytes into __m128i
             *   2. lo = bytes & 0x0F  (even-index nibbles: 0,2,4,6,8,10,12,14)
             *   3. hi = bytes >> 4     (odd-index nibbles:  1,3,5,7,9,11,13,15)
             *   4. interleaved = unpacklo_epi8(lo, hi)  → [n0,n1,n2,n3,...,n15]
             *   5. cvtepu8_epi32 lower 8 → __m256i → cvtepi32_ps → __m256 float
             *   6. cvtepu8_epi32 upper 8 → __m256i → cvtepi32_ps → __m256 float
             */
            for (; k + 31 < count; k += 32) {
                /* --- First 16 nibbles (8 packed bytes) --- */
                __m128i raw0   = _mm_loadl_epi64((const __m128i*)(packed + k / 2));
                __m128i lo0    = _mm_and_si128(raw0, lo_mask_128);
                __m128i hi0    = _mm_and_si128(_mm_srli_epi16(raw0, 4), lo_mask_128);
                __m128i nib0   = _mm_unpacklo_epi8(lo0, hi0);  /* 16 nibbles in order */

                __m256i i32_a  = _mm256_cvtepu8_epi32(nib0);
                __m256 fa      = _mm256_cvtepi32_ps(i32_a);
                __m256 inp_a   = _mm256_loadu_ps(inp + k);
                dot_acc0       = _mm256_fmadd_ps(fa, inp_a, dot_acc0);

                __m256i i32_b  = _mm256_cvtepu8_epi32(_mm_srli_si128(nib0, 8));
                __m256 fb      = _mm256_cvtepi32_ps(i32_b);
                __m256 inp_b   = _mm256_loadu_ps(inp + k + 8);
                dot_acc1       = _mm256_fmadd_ps(fb, inp_b, dot_acc1);

                /* --- Second 16 nibbles (next 8 packed bytes) --- */
                __m128i raw1   = _mm_loadl_epi64((const __m128i*)(packed + k / 2 + 8));
                __m128i lo1    = _mm_and_si128(raw1, lo_mask_128);
                __m128i hi1    = _mm_and_si128(_mm_srli_epi16(raw1, 4), lo_mask_128);
                __m128i nib1   = _mm_unpacklo_epi8(lo1, hi1);

                __m256i i32_c  = _mm256_cvtepu8_epi32(nib1);
                __m256 fc      = _mm256_cvtepi32_ps(i32_c);
                __m256 inp_c   = _mm256_loadu_ps(inp + k + 16);
                dot_acc2       = _mm256_fmadd_ps(fc, inp_c, dot_acc2);

                __m256i i32_d  = _mm256_cvtepu8_epi32(_mm_srli_si128(nib1, 8));
                __m256 fd      = _mm256_cvtepi32_ps(i32_d);
                __m256 inp_d   = _mm256_loadu_ps(inp + k + 24);
                dot_acc3       = _mm256_fmadd_ps(fd, inp_d, dot_acc3);
            }

            /* Process remaining 16 nibbles (8 packed bytes) */
            for (; k + 15 < count; k += 16) {
                __m128i raw    = _mm_loadl_epi64((const __m128i*)(packed + k / 2));
                __m128i lo     = _mm_and_si128(raw, lo_mask_128);
                __m128i hi     = _mm_and_si128(_mm_srli_epi16(raw, 4), lo_mask_128);
                __m128i nib    = _mm_unpacklo_epi8(lo, hi);

                __m256i i32_a  = _mm256_cvtepu8_epi32(nib);
                __m256 fa      = _mm256_cvtepi32_ps(i32_a);
                dot_acc0       = _mm256_fmadd_ps(fa, _mm256_loadu_ps(inp + k), dot_acc0);

                __m256i i32_b  = _mm256_cvtepu8_epi32(_mm_srli_si128(nib, 8));
                __m256 fb      = _mm256_cvtepi32_ps(i32_b);
                dot_acc1       = _mm256_fmadd_ps(fb, _mm256_loadu_ps(inp + k + 8), dot_acc1);
            }

            /* Scalar tail (< 16 remaining nibbles) */
            float tail_dot = 0.0f;
            for (; k < count; k++) {
                int byte_idx = k / 2;
                uint8_t q = (k % 2 == 0) ? (packed[byte_idx] & 0x0F)
                                          : ((packed[byte_idx] >> 4) & 0x0F);
                tail_dot += (float)q * inp[k];
            }

            /* ── Reduce: scale * dot(nib, inp) + zero * sum(inp) ─ */
            __m256 combined = _mm256_add_ps(
                _mm256_add_ps(dot_acc0, dot_acc1),
                _mm256_add_ps(dot_acc2, dot_acc3)
            );
            float nib_dot = hsum_avx(combined) + tail_dot;

            /* Factored affine: scale * Σ(nib*inp) + zero * Σ(inp) */
            float block_result = scale * nib_dot + zero * input_block_sums[b];
            row_acc = _mm256_add_ps(row_acc,
                        _mm256_set_ps(0,0,0,0, 0,0,0, block_result));

            col += bs;
            wp += block_bytes;
        }

        out[r] = hsum_avx(row_acc);
    }

    free(input_block_sums);
}

/* ── AVX2 fused dequant-matvec: 4-bit symmetric ─────────────────── *
 *
 * Symmetric layout: [FP16 scale (2B)] [2B reserved] [packed nibbles]
 * Dequant: value = scale * (nibble - 8)
 *
 * Factored: out[r] = Σ_b [ scale_b · (dot(nib, inp) - 8 · sum(inp)) ]
 */
void qsf_matvec_4bit_sym_avx2(const void* w, const float* in, float* out,
                               int rows, int cols, int bs) {
    if (rows <= 0 || cols <= 0) return;

    const size_t block_bytes = qsf_quant_block_size(QSF_QUANT_4BIT_SYM, bs);
    const int num_blocks_per_row = (cols + bs - 1) / bs;
    const __m128i lo_mask_128 = _mm_set1_epi8(0x0F);

    /* Precompute input block sums */
    float* input_block_sums = (float*)malloc(num_blocks_per_row * sizeof(float));
    if (!input_block_sums) {
        qsf_matvec_4bit_sym_scalar(w, in, out, rows, cols, bs);
        return;
    }
    {
        int col = 0;
        for (int b = 0; b < num_blocks_per_row; b++) {
            int count = (col + bs <= cols) ? bs : (cols - col);
            input_block_sums[b] = block_sum_avx(in + col, count);
            col += bs;
        }
    }

    const uint8_t* wp = (const uint8_t*)w;

    for (int r = 0; r < rows; r++) {
        float row_result = 0.0f;
        int col = 0;

        for (int b = 0; b < num_blocks_per_row; b++) {
            int count = (col + bs <= cols) ? bs : (cols - col);

            /* Decode scale via F16C (zero is unused in symmetric) */
            uint16_t scale_h;
            memcpy(&scale_h, wp, 2);
            __m128i hval = _mm_set_epi16(0,0,0,0, 0,0,0, scale_h);
            float scale  = _mm_cvtss_f32(_mm_cvtph_ps(hval));

            const uint8_t* packed = wp + 4;
            __m256 dot_acc0 = _mm256_setzero_ps();
            __m256 dot_acc1 = _mm256_setzero_ps();
            const float* inp = in + col;
            int k = 0;

            for (; k + 15 < count; k += 16) {
                __m128i raw  = _mm_loadl_epi64((const __m128i*)(packed + k / 2));
                __m128i lo   = _mm_and_si128(raw, lo_mask_128);
                __m128i hi   = _mm_and_si128(_mm_srli_epi16(raw, 4), lo_mask_128);
                __m128i nib  = _mm_unpacklo_epi8(lo, hi);

                __m256i i32a = _mm256_cvtepu8_epi32(nib);
                __m256 fa    = _mm256_cvtepi32_ps(i32a);
                dot_acc0     = _mm256_fmadd_ps(fa, _mm256_loadu_ps(inp + k), dot_acc0);

                __m256i i32b = _mm256_cvtepu8_epi32(_mm_srli_si128(nib, 8));
                __m256 fb    = _mm256_cvtepi32_ps(i32b);
                dot_acc1     = _mm256_fmadd_ps(fb, _mm256_loadu_ps(inp + k + 8), dot_acc1);
            }

            float tail_dot = 0.0f;
            for (; k < count; k++) {
                int byte_idx = k / 2;
                uint8_t q = (k % 2 == 0) ? (packed[byte_idx] & 0x0F)
                                          : ((packed[byte_idx] >> 4) & 0x0F);
                tail_dot += (float)q * inp[k];
            }

            float nib_dot = hsum_avx(_mm256_add_ps(dot_acc0, dot_acc1)) + tail_dot;
            /* Symmetric: value = scale * (nib - 8), factored: scale*(dot - 8*sum) */
            row_result += scale * (nib_dot - 8.0f * input_block_sums[b]);

            col += bs;
            wp += block_bytes;
        }

        out[r] = row_result;
    }

    free(input_block_sums);
}

/* ── AVX2 fused dequant-matvec: 2-bit ───────────────────────────── *
 *
 * Block layout: [FP16 scale (2B)] [FP16 zero (2B)] [packed 2-bit (bs/4 B)]
 * Dequant: value = scale * crumb + zero   (crumb ∈ {0,1,2,3})
 *
 * Unpack strategy: extract 4 crumbs per byte via successive shift+mask,
 * then widen to float for FMA.
 */
void qsf_matvec_2bit_avx2(const void* w, const float* in, float* out,
                           int rows, int cols, int bs) {
    if (rows <= 0 || cols <= 0) return;

    const size_t block_bytes = qsf_quant_block_size(QSF_QUANT_2BIT_ASYM, bs);
    const int num_blocks_per_row = (cols + bs - 1) / bs;

    /* Precompute input block sums */
    float* input_block_sums = (float*)malloc(num_blocks_per_row * sizeof(float));
    if (!input_block_sums) {
        qsf_matvec_2bit_scalar(w, in, out, rows, cols, bs);
        return;
    }
    {
        int col = 0;
        for (int b = 0; b < num_blocks_per_row; b++) {
            int count = (col + bs <= cols) ? bs : (cols - col);
            input_block_sums[b] = block_sum_avx(in + col, count);
            col += bs;
        }
    }

    const uint8_t* wp = (const uint8_t*)w;

    for (int r = 0; r < rows; r++) {
        float row_result = 0.0f;
        int col = 0;

        for (int b = 0; b < num_blocks_per_row; b++) {
            int count = (col + bs <= cols) ? bs : (cols - col);

            uint16_t scale_h, zero_h;
            memcpy(&scale_h, wp, 2);
            memcpy(&zero_h, wp + 2, 2);
            __m128i hpair = _mm_set_epi16(0,0,0,0, 0,0, zero_h, scale_h);
            __m128 fpair  = _mm_cvtph_ps(hpair);
            float scale = _mm_cvtss_f32(fpair);
            float zero  = _mm_cvtss_f32(_mm_shuffle_ps(fpair, fpair, 0x55));

            const uint8_t* packed = wp + 4;
            __m256 dot_acc0 = _mm256_setzero_ps();
            __m256 dot_acc1 = _mm256_setzero_ps();
            const float* inp = in + col;
            int k = 0;

            /* Process 8 crumbs (2 bytes) at a time → 1 × __m256 FMA
             * Byte layout: each byte has 4 crumbs at bits [1:0], [3:2], [5:4], [7:6]
             * Two bytes → 8 crumbs → 8 floats → 1 __m256 FMA */
            for (; k + 15 < count; k += 16) {
                /* First 8 crumbs from 2 bytes */
                uint32_t raw0;
                memcpy(&raw0, packed + k / 4, 2);  /* 2 bytes = 8 crumbs */
                /* Unpack: spread each 2-bit crumb into a 32-bit lane */
                __m256i bits0 = _mm256_set_epi32(
                    (raw0 >> 14) & 3, (raw0 >> 12) & 3,
                    (raw0 >> 10) & 3, (raw0 >>  8) & 3,
                    (raw0 >>  6) & 3, (raw0 >>  4) & 3,
                    (raw0 >>  2) & 3, (raw0 >>  0) & 3
                );
                __m256 f0 = _mm256_cvtepi32_ps(bits0);
                dot_acc0  = _mm256_fmadd_ps(f0, _mm256_loadu_ps(inp + k), dot_acc0);

                /* Second 8 crumbs from next 2 bytes */
                uint32_t raw1;
                memcpy(&raw1, packed + k / 4 + 2, 2);
                __m256i bits1 = _mm256_set_epi32(
                    (raw1 >> 14) & 3, (raw1 >> 12) & 3,
                    (raw1 >> 10) & 3, (raw1 >>  8) & 3,
                    (raw1 >>  6) & 3, (raw1 >>  4) & 3,
                    (raw1 >>  2) & 3, (raw1 >>  0) & 3
                );
                __m256 f1 = _mm256_cvtepi32_ps(bits1);
                dot_acc1  = _mm256_fmadd_ps(f1, _mm256_loadu_ps(inp + k + 8), dot_acc1);
            }

            /* Scalar tail */
            float tail_dot = 0.0f;
            for (; k < count; k++) {
                int byte_idx = k / 4;
                int bit_off  = (k % 4) * 2;
                uint8_t q = (packed[byte_idx] >> bit_off) & 0x03;
                tail_dot += (float)q * inp[k];
            }

            float nib_dot = hsum_avx(_mm256_add_ps(dot_acc0, dot_acc1)) + tail_dot;
            row_result += scale * nib_dot + zero * input_block_sums[b];

            col += bs;
            wp += block_bytes;
        }

        out[r] = row_result;
    }

    free(input_block_sums);
}

/* ── AVX2 fused dequant-matvec: Outlier-aware 2-bit ──────────────── */
void qsf_matvec_outlier_2bit_avx2(const void* w, const float* in, float* out,
                                   int rows, int cols, int bs) {
    const uint8_t* p = (const uint8_t*)w;

    /* 1. Read outlier header */
    uint32_t num_outliers;
    memcpy(&num_outliers, p, 4);
    p += 4;

    const uint8_t* outlier_entries = p;
    p += (size_t)num_outliers * 6;

    /* 2. Run standard AVX2 2-bit matvec on base blocks */
    qsf_matvec_2bit_avx2(p, in, out, rows, cols, bs);

    /* 3. Sparse outlier correction */
    /* AVX2 optimization: not worth vectorizing sparse random access.
     * But we can batch the FP16->FP32 conversion using F16C. */
    const uint8_t* entry = outlier_entries;
    uint32_t i = 0;

    /* Process batches of 8 outliers for F16C conversion */
    for (; i + 7 < num_outliers; i += 8) {
        uint32_t flat_indices[8];
        uint16_t fp16_vals[8];

        /* Gather 8 entries */
        for (int k = 0; k < 8; k++) {
            memcpy(&flat_indices[k], entry, 4);
            memcpy(&fp16_vals[k], entry + 4, 2);
            entry += 6;
        }

        /* Convert 8 FP16 values at once */
        __m128i vh = _mm_loadu_si128((const __m128i*)fp16_vals);
        __m256 vf  = _mm256_cvtph_ps(vh);
        float vals[8];
        _mm256_storeu_ps(vals, vf);

        /* Apply corrections */
        for (int k = 0; k < 8; k++) {
            uint32_t flat_idx = flat_indices[k];
            int r = (int)(flat_idx / (uint32_t)cols);
            int c = (int)(flat_idx % (uint32_t)cols);
            if (r < rows && c < cols) {
                out[r] += vals[k] * in[c];
            }
        }
    }

    /* Scalar tail */
    for (; i < num_outliers; i++) {
        uint32_t flat_idx;
        uint16_t fp16_val;
        memcpy(&flat_idx, entry, 4);
        memcpy(&fp16_val, entry + 4, 2);
        entry += 6;

        int r = (int)(flat_idx / (uint32_t)cols);
        int c = (int)(flat_idx % (uint32_t)cols);
        if (r < rows && c < cols) {
            /* For single values, scalar conversion is fine or use intrinsic */
            /* Use intrinsic for consistency if F16C is available (checked in kernels.c) */
            __m128i vh = _mm_set1_epi16(fp16_val);
            __m128  vf = _mm_cvtph_ps(vh);
            float val  = _mm_cvtss_f32(vf);
            out[r] += val * in[c];
        }
    }
}

/* ── AVX2 fused dequant-matvec: Outlier-aware 4-bit ──────────────── */
void qsf_matvec_outlier_4bit_avx2(const void* w, const float* in, float* out,
                                   int rows, int cols, int bs) {
    const uint8_t* p = (const uint8_t*)w;

    uint32_t num_outliers;
    memcpy(&num_outliers, p, 4);
    p += 4;

    const uint8_t* outlier_entries = p;
    p += (size_t)num_outliers * 6;

    /* Run standard AVX2 4-bit matvec */
    qsf_matvec_4bit_avx2(p, in, out, rows, cols, bs);

    /* Sparse outlier correction (identical to 2-bit case) */
    const uint8_t* entry = outlier_entries;
    uint32_t i = 0;

    for (; i + 7 < num_outliers; i += 8) {
        uint32_t flat_indices[8];
        uint16_t fp16_vals[8];
        for (int k = 0; k < 8; k++) {
            memcpy(&flat_indices[k], entry, 4);
            memcpy(&fp16_vals[k], entry + 4, 2);
            entry += 6;
        }
        __m128i vh = _mm_loadu_si128((const __m128i*)fp16_vals);
        __m256 vf  = _mm256_cvtph_ps(vh);
        float vals[8];
        _mm256_storeu_ps(vals, vf);

        for (int k = 0; k < 8; k++) {
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
            __m128i vh = _mm_set1_epi16(fp16_val);
            __m128  vf = _mm_cvtph_ps(vh);
            float val  = _mm_cvtss_f32(vf);
            out[r] += val * in[c];
        }
    }
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
