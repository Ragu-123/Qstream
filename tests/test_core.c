/*
 * QStream - test_core.c
 * Unit tests for core QSF components.
 *
 * Test framework: minimal assert-based (no external deps).
 * Build: cmake --build build --target qstream_tests
 * Run:   ./build/Release/qstream_tests
 */
#include "qsf/qsf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  [TEST] %-40s ", #name); \
    fflush(stdout); \
} while(0)

#define PASS() do { \
    tests_passed++; \
    printf("PASS\n"); \
} while(0)

#define FAIL(msg) do { \
    tests_failed++; \
    printf("FAIL: %s\n", msg); \
} while(0)

#define ASSERT_NEAR(a, b, eps) \
    if (fabsf((a) - (b)) > (eps)) { \
        char _buf[128]; \
        snprintf(_buf, sizeof(_buf), "expected %.6f, got %.6f", (double)(b), (double)(a)); \
        FAIL(_buf); return; \
    }

#define ASSERT_EQ_INT(a, b) \
    if ((a) != (b)) { \
        char _buf[128]; \
        snprintf(_buf, sizeof(_buf), "expected %d, got %d", (int)(b), (int)(a)); \
        FAIL(_buf); return; \
    }

#define ASSERT_TRUE(cond) \
    if (!(cond)) { \
        FAIL(#cond " is false"); return; \
    }

/* ═══════════════════════════════════════════════════════════════════
 *  CRC-32 Tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_crc32_empty(void) {
    TEST(crc32_empty);
    uint32_t c = qsf_crc32("", 0);
    ASSERT_EQ_INT(c, 0x00000000);
    PASS();
}

static void test_crc32_known(void) {
    TEST(crc32_known);
    /* CRC32("123456789") = 0xCBF43926 */
    uint32_t c = qsf_crc32("123456789", 9);
    ASSERT_EQ_INT(c, 0xCBF43926);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 *  Error System Tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_error_strings(void) {
    TEST(error_strings);
    const char* s = qsf_error_string(QSF_OK);
    ASSERT_TRUE(s != NULL);
    ASSERT_TRUE(strcmp(s, "OK") == 0 || strcmp(s, "no error") == 0 ||
                strstr(s, "ok") != NULL || strstr(s, "OK") != NULL ||
                strstr(s, "success") != NULL);

    s = qsf_error_string(QSF_ERR_OUT_OF_MEMORY);
    ASSERT_TRUE(s != NULL);
    ASSERT_TRUE(strlen(s) > 0);
    PASS();
}

static void test_error_set_get(void) {
    TEST(error_set_get);
    qsf_set_error(QSF_ERR_INTERNAL, "test detail");
    ASSERT_EQ_INT(qsf_get_error(), QSF_ERR_INTERNAL);
    const char* detail = qsf_get_error_detail();
    ASSERT_TRUE(detail != NULL);
    ASSERT_TRUE(strstr(detail, "test") != NULL);
    qsf_clear_error();
    ASSERT_EQ_INT(qsf_get_error(), QSF_OK);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 *  Platform Detection Tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_platform_detect(void) {
    TEST(platform_detect);
    QSFPlatformInfo info;
    qsf_detect_platform(&info);

    /* Must detect at least SSE4.2 on any modern x86_64 */
#if defined(_M_X64) || defined(__x86_64__)
    ASSERT_TRUE(info.has_sse42);
#endif

    /* RAM must be nonzero */
    ASSERT_TRUE(info.total_ram_bytes > 0);

    /* Best ISA must be set */
    ASSERT_TRUE(info.best_isa > 0);

    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 *  Arena Allocator Tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_arena_basic(void) {
    TEST(arena_basic);
    Arena* a = arena_create(4096);
    ASSERT_TRUE(a != NULL);
    ASSERT_TRUE(a->base != NULL);
    ASSERT_TRUE(a->total_size >= 4096);

    /* Allocate */
    void* p1 = arena_alloc(a, 128, 16, "test1");
    ASSERT_TRUE(p1 != NULL);
    ASSERT_TRUE(((size_t)p1 & 15) == 0);  /* aligned */

    void* p2 = arena_alloc(a, 64, 64, "test2");
    ASSERT_TRUE(p2 != NULL);
    ASSERT_TRUE(((size_t)p2 & 63) == 0);  /* 64-byte aligned */
    ASSERT_TRUE(p2 != p1);  /* different pointers */

    /* Reset */
    arena_reset(a);
    void* p3 = arena_alloc(a, 128, 16, "test3");
    ASSERT_TRUE(p3 == p1);  /* should reuse same base */

    arena_destroy(a);
    PASS();
}

static void test_arena_oom(void) {
    TEST(arena_oom);
    Arena* a = arena_create(1024);
    ASSERT_TRUE(a != NULL);

    /* Request more than available */
    void* p = arena_alloc(a, 2048, 16, "oom_test");
    ASSERT_TRUE(p == NULL);

    arena_destroy(a);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 *  FP16 Conversion Tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_fp16_roundtrip(void) {
    TEST(fp16_roundtrip);
    float values[] = {0.0f, 1.0f, -1.0f, 0.5f, 65504.0f, -65504.0f};
    int n = sizeof(values) / sizeof(float);

    for (int i = 0; i < n; i++) {
        uint16_t h = qsf_fp32_to_fp16(values[i]);
        float    f = qsf_fp16_to_fp32(h);
        ASSERT_NEAR(f, values[i], 1.0f);  /* FP16 has limited precision */
    }
    PASS();
}

static void test_fp16_zero(void) {
    TEST(fp16_zero);
    uint16_t h = qsf_fp32_to_fp16(0.0f);
    float f = qsf_fp16_to_fp32(h);
    ASSERT_NEAR(f, 0.0f, 1e-10f);
    PASS();
}

static void test_fp16_bulk(void) {
    TEST(fp16_bulk);
    uint16_t in[8];
    float out[8];
    float expected[] = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, -0.25f, 100.0f, -100.0f};

    for (int i = 0; i < 8; i++) {
        in[i] = qsf_fp32_to_fp16(expected[i]);
    }
    qsf_fp16_to_fp32_array(in, out, 8);

    for (int i = 0; i < 8; i++) {
        ASSERT_NEAR(out[i], expected[i], 0.2f);
    }
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 *  Kernel Tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_dot_product(void) {
    TEST(dot_product);
    float a[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[] = {8, 7, 6, 5, 4, 3, 2, 1};
    /* 8+14+18+20+20+18+14+8 = 120 */
    float result = qsf_dot_scalar(a, b, 8);
    ASSERT_NEAR(result, 120.0f, 1e-4f);
    PASS();
}

static void test_dot_product_size0(void) {
    TEST(dot_product_size0);
    float a[] = {1, 2, 3};
    float b[] = {4, 5, 6};
    float result = qsf_dot_scalar(a, b, 0);
    ASSERT_NEAR(result, 0.0f, 1e-6f);
    PASS();
}

static void test_vec_add(void) {
    TEST(vec_add);
    float a[] = {1, 2, 3, 4};
    float b[] = {10, 20, 30, 40};
    float out[4];
    qsf_vec_add_scalar_impl(a, b, out, 4);
    ASSERT_NEAR(out[0], 11.0f, 1e-6f);
    ASSERT_NEAR(out[1], 22.0f, 1e-6f);
    ASSERT_NEAR(out[2], 33.0f, 1e-6f);
    ASSERT_NEAR(out[3], 44.0f, 1e-6f);
    PASS();
}

static void test_softmax(void) {
    TEST(softmax);
    float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4];
    qsf_softmax_scalar(in, out, 4);

    /* Sum must be 1.0 */
    float sum = out[0] + out[1] + out[2] + out[3];
    ASSERT_NEAR(sum, 1.0f, 1e-5f);

    /* Output must be monotonically increasing */
    ASSERT_TRUE(out[0] < out[1]);
    ASSERT_TRUE(out[1] < out[2]);
    ASSERT_TRUE(out[2] < out[3]);
    PASS();
}

static void test_softmax_size0(void) {
    TEST(softmax_size0);
    /* Should not crash */
    qsf_softmax_scalar(NULL, NULL, 0);
    PASS();
}

static void test_silu(void) {
    TEST(silu);
    float x[] = {0.0f, 1.0f, -1.0f, 5.0f};
    float orig[] = {0.0f, 1.0f, -1.0f, 5.0f};
    qsf_silu_scalar(x, 4);

    /* SiLU(0) = 0 */
    ASSERT_NEAR(x[0], 0.0f, 1e-6f);
    /* SiLU(x) = x * sigmoid(x) */
    ASSERT_NEAR(x[1], 1.0f / (1.0f + expf(-1.0f)), 1e-5f);
    /* SiLU(-1) = -1 * sigmoid(-1) */
    ASSERT_NEAR(x[2], -1.0f / (1.0f + expf(1.0f)), 1e-5f);
    PASS();
}

static void test_rms_norm(void) {
    TEST(rms_norm);
    float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[4]  = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];

    qsf_rms_norm_scalar(in, out, w, NULL, 4, 1e-5f);

    /* RMS = sqrt((1+4+9+16)/4 + eps) = sqrt(7.5 + eps) */
    float rms = sqrtf(30.0f / 4.0f + 1e-5f);
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(out[i], in[i] / rms, 1e-4f);
    }
    PASS();
}

static void test_layer_norm(void) {
    TEST(layer_norm);
    float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[4]  = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];

    qsf_layer_norm_scalar(in, out, w, NULL, 4, 1e-5f);

    /* Mean = 2.5, Var = 1.25, output should be zero-mean */
    float sum = out[0] + out[1] + out[2] + out[3];
    ASSERT_NEAR(sum, 0.0f, 1e-4f);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 *  Utility Kernel Tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_find_min_max(void) {
    TEST(find_min_max);
    float data[] = {3.0f, -1.0f, 7.0f, 2.0f, 0.0f};
    float lo, hi;
    qsf_find_min_max(data, 5, &lo, &hi);
    ASSERT_NEAR(lo, -1.0f, 1e-6f);
    ASSERT_NEAR(hi, 7.0f, 1e-6f);
    PASS();
}

static void test_find_min_max_size0(void) {
    TEST(find_min_max_size0);
    float lo = 999, hi = 999;
    qsf_find_min_max(NULL, 0, &lo, &hi);
    ASSERT_NEAR(lo, 0.0f, 1e-6f);
    ASSERT_NEAR(hi, 0.0f, 1e-6f);
    PASS();
}

static void test_sum_float(void) {
    TEST(sum_float);
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float sum = qsf_sum_float(data, 5);
    ASSERT_NEAR(sum, 15.0f, 1e-5f);
    PASS();
}

static void test_memset_pattern(void) {
    TEST(memset_pattern);
    float buf[8];
    qsf_memset_pattern(buf, -HUGE_VALF, 8);  /* -INFINITY */
    for (int i = 0; i < 8; i++) {
        ASSERT_TRUE(buf[i] < -1e30f);
    }
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 *  Quantization Tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_quant_block_size(void) {
    TEST(quant_block_size);
    /* 4-bit asym: 4 bytes header + block_size/2 data bytes */
    size_t s = qsf_quant_block_size(QSF_QUANT_4BIT_ASYM, 64);
    ASSERT_TRUE(s > 0);

    /* 2-bit: 4 bytes header + block_size/4 data bytes */
    s = qsf_quant_block_size(QSF_QUANT_2BIT_ASYM, 64);
    ASSERT_TRUE(s > 0);
    PASS();
}

static void test_quant_4bit_roundtrip(void) {
    TEST(quant_4bit_roundtrip);
    /* Create a simple block of values */
    float original[64];
    for (int i = 0; i < 64; i++) {
        original[i] = (float)i / 64.0f;
    }

    /* Quantize */
    size_t block_sz = qsf_quant_block_size(QSF_QUANT_4BIT_ASYM, 64);
    uint8_t* packed = (uint8_t*)malloc(block_sz);
    ASSERT_TRUE(packed != NULL);

    qsf_quant_block_4bit(original, packed, 64);

    /* Dequantize */
    float restored[64];
    qsf_dequant_block_4bit(packed, restored, 64);

    /* Check: should be somewhat close (4-bit has ~6% error) */
    float max_err = 0;
    for (int i = 0; i < 64; i++) {
        float err = fabsf(original[i] - restored[i]);
        if (err > max_err) max_err = err;
    }
    /* 4-bit with range [0, ~1]: max step is range/15 ≈ 0.067 */
    ASSERT_TRUE(max_err < 0.15f);

    free(packed);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 *  Kernel Dispatch Tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_kernel_dispatch(void) {
    TEST(kernel_dispatch);
    QSFPlatformInfo pi;
    qsf_detect_platform(&pi);

    QSFKernelTable kt;
    qsf_kernels_init(&kt, &pi);

    /* All function pointers must be non-NULL */
    ASSERT_TRUE(kt.matvec_2bit != NULL);
    ASSERT_TRUE(kt.matvec_3bit != NULL);
    ASSERT_TRUE(kt.matvec_4bit != NULL);
    ASSERT_TRUE(kt.vec_add != NULL);
    ASSERT_TRUE(kt.vec_mul != NULL);
    ASSERT_TRUE(kt.silu != NULL);
    ASSERT_TRUE(kt.gelu != NULL);
    ASSERT_TRUE(kt.relu != NULL);
    ASSERT_TRUE(kt.softmax != NULL);
    ASSERT_TRUE(kt.dot != NULL);
    ASSERT_TRUE(kt.rms_norm != NULL);
    ASSERT_TRUE(kt.layer_norm != NULL);
    ASSERT_TRUE(kt.fp16_to_fp32 != NULL);

    /* Test dispatched dot product */
    float a[] = {1, 2, 3, 4};
    float b[] = {4, 3, 2, 1};
    float d = kt.dot(a, b, 4);
    ASSERT_NEAR(d, 20.0f, 1e-4f);

    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 *  Sampling Tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_sampling_rng(void) {
    TEST(sampling_rng);
    QSFRng rng;
    qsf_rng_seed(&rng, 42);

    /* Should produce different values */
    float v1 = qsf_rng_float(&rng);
    float v2 = qsf_rng_float(&rng);
    ASSERT_TRUE(v1 >= 0.0f && v1 < 1.0f);
    ASSERT_TRUE(v2 >= 0.0f && v2 < 1.0f);
    ASSERT_TRUE(v1 != v2);  /* extremely unlikely */
    PASS();
}

static void test_sampling_rng_deterministic(void) {
    TEST(sampling_rng_deterministic);
    QSFRng s1, s2;
    qsf_rng_seed(&s1, 123);
    qsf_rng_seed(&s2, 123);

    for (int i = 0; i < 100; i++) {
        float a = qsf_rng_float(&s1);
        float b = qsf_rng_float(&s2);
        ASSERT_NEAR(a, b, 1e-10f);
    }
    PASS();
}

static void test_sampling_argmax(void) {
    TEST(sampling_argmax);
    float logits[] = {0.1f, 0.5f, 0.3f, 0.9f, 0.2f};
    int idx = qsf_argmax(logits, 5);
    ASSERT_EQ_INT(idx, 3);
    PASS();
}

static void test_sampling_argmax_tie(void) {
    TEST(sampling_argmax_tie);
    /* Tie: should return lowest index (§4.3 deterministic tie-breaking) */
    float logits[] = {0.5f, 0.5f, 0.5f};
    int idx = qsf_argmax(logits, 3);
    ASSERT_EQ_INT(idx, 0);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 *  Header Validation Tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_header_valid(void) {
    TEST(header_valid);
    QSFHeader h;
    memset(&h, 0, sizeof(h));
    h.magic = QSF_MAGIC;
    h.version = QSF_VERSION;
    h.header_size = QSF_HEADER_SIZE;
    h.num_layers = 12;
    h.hidden_dim = 768;
    h.num_heads = 12;
    h.num_kv_heads = 12;
    h.vocab_size = 50257;
    h.max_seq_len = 1024;
    h.intermediate_dim = 3072;
    h.head_dim = 64;
    h.block_size = 64;

    QSFError err = qsf_validate_header(&h);
    ASSERT_EQ_INT(err, QSF_OK);
    PASS();
}

static void test_header_bad_magic(void) {
    TEST(header_bad_magic);
    QSFHeader h;
    memset(&h, 0, sizeof(h));
    h.magic = 0xDEADBEEF;
    QSFError err = qsf_validate_header(&h);
    ASSERT_TRUE(err != QSF_OK);
    PASS();
}

static void test_header_zero_layers(void) {
    TEST(header_zero_layers);
    QSFHeader h;
    memset(&h, 0, sizeof(h));
    h.magic = QSF_MAGIC;
    h.version = QSF_VERSION;
    h.header_size = QSF_HEADER_SIZE;
    h.num_layers = 0;
    QSFError err = qsf_validate_header(&h);
    ASSERT_TRUE(err != QSF_OK);
    PASS();
}

static void test_quant_2bit_roundtrip(void) {
    TEST(quant_2bit_roundtrip);
    float input[64];
    float output[64];
    for (int i = 0; i < 64; i++) input[i] = (float)i / 64.0f;

    uint8_t block[256];
    qsf_quant_block_2bit(input, block, 64);
    qsf_dequant_block_2bit(block, output, 64);

    /* 2-bit only has 4 levels, so tolerance is wider */
    for (int i = 0; i < 64; i++) {
        ASSERT_TRUE(fabs(output[i] - input[i]) < 0.25f);
    }
    PASS();
}

static void test_quant_3bit_roundtrip(void) {
    TEST(quant_3bit_roundtrip);
    float input[64];
    float output[64];
    for (int i = 0; i < 64; i++) input[i] = (float)i / 64.0f;

    uint8_t block[256];
    qsf_quant_block_3bit(input, block, 64);
    qsf_dequant_block_3bit(block, output, 64);

    /* 3-bit has 8 levels -> tolerance ~0.125 */
    for (int i = 0; i < 64; i++) {
        ASSERT_TRUE(fabs(output[i] - input[i]) < 0.15f);
    }
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 *  Repeat Penalty Tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_repeat_penalty(void) {
    TEST(repeat_penalty);
    float logits[] = {1.0f, 2.0f, 3.0f, -1.0f, 0.5f};
    uint32_t recent[] = {1, 3};  /* penalize tokens 1 and 3 */

    qsf_apply_repeat_penalty(logits, 5, recent, 2, 2.0f);

    /* Token 1 was positive 2.0 -> should be divided by 2.0 = 1.0 */
    ASSERT_NEAR(logits[1], 1.0f, 1e-5f);
    /* Token 3 was negative -1.0 -> should be multiplied by 2.0 = -2.0 */
    ASSERT_NEAR(logits[3], -2.0f, 1e-5f);
    /* Tokens 0, 2, 4 should be unchanged */
    ASSERT_NEAR(logits[0], 1.0f, 1e-5f);
    ASSERT_NEAR(logits[2], 3.0f, 1e-5f);
    ASSERT_NEAR(logits[4], 0.5f, 1e-5f);
    PASS();
}

static void test_repeat_penalty_noop(void) {
    TEST(repeat_penalty_noop);
    float logits[] = {1.0f, 2.0f};
    /* Penalty <= 1.0 should be a no-op */
    qsf_apply_repeat_penalty(logits, 2, NULL, 0, 1.0f);
    ASSERT_NEAR(logits[0], 1.0f, 1e-5f);
    ASSERT_NEAR(logits[1], 2.0f, 1e-5f);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 *  Struct Size Verification Tests
 * ═══════════════════════════════════════════════════════════════════ */
static void test_struct_sizes(void) {
    TEST(struct_sizes);
    ASSERT_EQ_INT((int)sizeof(QSFHeader), 256);
    ASSERT_EQ_INT((int)sizeof(QSFLayerIndex), 48);
    ASSERT_EQ_INT((int)sizeof(QSFTensorHeader), 24);
    ASSERT_EQ_INT((int)sizeof(QSFEmbeddingHeader), 32);
    ASSERT_EQ_INT((int)sizeof(QSFTokenizerHeader), 32);
    ASSERT_EQ_INT((int)sizeof(QSFFinalHeader), 16);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════════
 *  Main
 * ═══════════════════════════════════════════════════════════════════ */
int main(void) {
    printf("\n========================================\n");
    printf("  QStream Core Unit Tests\n");
    printf("========================================\n\n");

    printf("[CRC-32]\n");
    test_crc32_empty();
    test_crc32_known();

    printf("\n[Error System]\n");
    test_error_strings();
    test_error_set_get();

    printf("\n[Platform]\n");
    test_platform_detect();

    printf("\n[Arena]\n");
    test_arena_basic();
    test_arena_oom();

    printf("\n[FP16]\n");
    test_fp16_roundtrip();
    test_fp16_zero();
    test_fp16_bulk();

    printf("\n[Kernels - Scalar]\n");
    test_dot_product();
    test_dot_product_size0();
    test_vec_add();
    test_softmax();
    test_softmax_size0();
    test_silu();
    test_rms_norm();
    test_layer_norm();

    printf("\n[Utility Kernels]\n");
    test_find_min_max();
    test_find_min_max_size0();
    test_sum_float();
    test_memset_pattern();

    printf("\n[Quantization]\n");
    test_quant_block_size();
    test_quant_4bit_roundtrip();
    test_quant_2bit_roundtrip();
    test_quant_3bit_roundtrip();

    printf("\n[Kernel Dispatch]\n");
    test_kernel_dispatch();

    printf("\n[Sampling]\n");
    test_sampling_rng();
    test_sampling_rng_deterministic();
    test_sampling_argmax();
    test_sampling_argmax_tie();
    test_repeat_penalty();
    test_repeat_penalty_noop();

    printf("\n[Header Validation]\n");
    test_header_valid();
    test_header_bad_magic();
    test_header_zero_layers();

    printf("\n[Struct Sizes]\n");
    test_struct_sizes();

    printf("\n========================================\n");
    printf("  Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0) {
        printf(", %d FAILED", tests_failed);
    }
    printf("\n========================================\n\n");

    return tests_failed > 0 ? 1 : 0;
}
