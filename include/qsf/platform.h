/*
 * QStream - platform.h
 * CPU feature detection, OS abstractions, endian utilities.
 */
#ifndef QSF_PLATFORM_H
#define QSF_PLATFORM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* x86 flags */
    int has_sse42;
    int has_avx;
    int has_avx2;
    int has_fma;
    int has_f16c;
    int has_avx512f;
    int has_avx512bw;
    int has_avx512vnni;

    /* ARM flags */
    int has_neon;
    int has_fp16_arith;

    /* System info */
    uint64_t total_ram_bytes;
    uint64_t available_ram_bytes;
    int      num_cores;
    int      is_little_endian;

    /* Detected ISA tier */
    enum {
        QSF_ISA_SCALAR = 0,
        QSF_ISA_SSE42  = 1,
        QSF_ISA_AVX2   = 2,
        QSF_ISA_AVX512 = 3,
        QSF_ISA_NEON   = 4,
    } best_isa;
} QSFPlatformInfo;

/* Detect and fill platform info (call once at startup) */
void     qsf_detect_platform(QSFPlatformInfo* info);

/* Query system RAM */
uint64_t qsf_get_total_ram(void);
uint64_t qsf_get_available_ram(void);

/* Flush denormals to zero for current thread */
void     qsf_flush_denormals(void);

/* Endian helpers */
static inline int qsf_is_little_endian(void) {
    uint16_t x = 1;
    return *(uint8_t*)&x == 1;
}

static inline uint16_t qsf_swap16(uint16_t v) {
    return (v >> 8) | (v << 8);
}

static inline uint32_t qsf_swap32(uint32_t v) {
    return ((v >> 24) & 0xFF) | ((v >> 8) & 0xFF00) |
           ((v << 8) & 0xFF0000) | ((v << 24) & 0xFF000000u);
}

static inline uint64_t qsf_swap64(uint64_t v) {
    return ((uint64_t)qsf_swap32((uint32_t)v) << 32) |
           qsf_swap32((uint32_t)(v >> 32));
}

#ifdef __cplusplus
}
#endif
#endif /* QSF_PLATFORM_H */
