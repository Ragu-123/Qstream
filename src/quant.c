/*
 * QStream - quant.c
 * Block-level quantization / dequantization for 2/3/4-bit with FP16 scale+zero.
 */
#include "qsf/quant.h"
#include <math.h>
#include <string.h>

/* ── FP16 ↔ FP32 conversions ────────────────────────────────────── */

float qsf_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {                      /* zero */
            uint32_t f = sign;
            float result;
            memcpy(&result, &f, 4);
            return result;
        }
        /* subnormal → normalize */
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++;
        mant &= 0x3FF;
    } else if (exp == 31) {                   /* inf / nan */
        uint32_t f = sign | 0x7F800000u | ((uint32_t)mant << 13);
        float result;
        memcpy(&result, &f, 4);
        return result;
    }

    exp = exp + 127 - 15;
    uint32_t f = sign | (exp << 23) | ((uint32_t)mant << 13);
    float result;
    memcpy(&result, &f, 4);
    return result;
}

uint16_t qsf_fp32_to_fp16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t  exp  = ((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u & 0x7FFFFF;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;  /* too small → zero */
        mant = (mant | 0x800000) >> (1 - exp);
        return (uint16_t)(sign | (mant >> 13));
    } else if (exp >= 31) {
        if (exp == 143 && mant) {              /* NaN */
            return (uint16_t)(sign | 0x7E00 | (mant >> 13));
        }
        return (uint16_t)(sign | 0x7C00);      /* inf */
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

void qsf_fp16_to_fp32_array(const uint16_t* in, float* out, int count) {
    for (int i = 0; i < count; i++) {
        out[i] = qsf_fp16_to_fp32(in[i]);
    }
}

/* ── Block sizes ─────────────────────────────────────────────────── */

size_t qsf_quant_block_size(QSFQuantType type, int block_size) {
    size_t meta = 4;  /* 2 bytes scale + 2 bytes zero (both fp16) */
    switch (type) {
        case QSF_QUANT_2BIT_ASYM:
        case QSF_QUANT_2BIT_SYM:
            return meta + (size_t)(block_size * 2 + 7) / 8;
        case QSF_QUANT_3BIT_ASYM:
            return meta + (size_t)(block_size * 3 + 7) / 8;
        case QSF_QUANT_4BIT_ASYM:
        case QSF_QUANT_4BIT_SYM:
            return meta + (size_t)(block_size * 4 + 7) / 8;
        case QSF_QUANT_FP16:
            return (size_t)block_size * 2;
        default:
            return 0;
    }
}

/* ── 2-bit dequantization ────────────────────────────────────────── */
/*
 * Block layout (block_size=64):
 *   [uint16 scale_fp16] [uint16 zero_fp16] [16 bytes packed 2-bit values]
 *   Total: 20 bytes
 *
 * Dequant: value = scale * quantized + zero
 */
void qsf_dequant_block_2bit(const void* block_data, float* out, int count) {
    const uint8_t* p = (const uint8_t*)block_data;
    uint16_t scale_h, zero_h;
    memcpy(&scale_h, p, 2);
    memcpy(&zero_h, p + 2, 2);
    float scale = qsf_fp16_to_fp32(scale_h);
    float zero  = qsf_fp16_to_fp32(zero_h);
    p += 4;

    for (int i = 0; i < count; i++) {
        int byte_idx = i / 4;
        int bit_off  = (i % 4) * 2;
        uint8_t q = (p[byte_idx] >> bit_off) & 0x03;
        out[i] = scale * (float)q + zero;
    }
}

/* ── 3-bit dequantization ────────────────────────────────────────── */
void qsf_dequant_block_3bit(const void* block_data, float* out, int count) {
    const uint8_t* p = (const uint8_t*)block_data;
    uint16_t scale_h, zero_h;
    memcpy(&scale_h, p, 2);
    memcpy(&zero_h, p + 2, 2);
    float scale = qsf_fp16_to_fp32(scale_h);
    float zero  = qsf_fp16_to_fp32(zero_h);
    p += 4;

    /* Extract 3-bit values from bit stream */
    int bit_pos = 0;
    for (int i = 0; i < count; i++) {
        int byte_idx = bit_pos / 8;
        int bit_off  = bit_pos % 8;
        uint16_t two_bytes;
        memcpy(&two_bytes, p + byte_idx, 2);  /* safe: always enough data */
        uint8_t q = (two_bytes >> bit_off) & 0x07;
        out[i] = scale * (float)q + zero;
        bit_pos += 3;
    }
}

/* ── 4-bit asymmetric dequantization ─────────────────────────────── */
void qsf_dequant_block_4bit(const void* block_data, float* out, int count) {
    const uint8_t* p = (const uint8_t*)block_data;
    uint16_t scale_h, zero_h;
    memcpy(&scale_h, p, 2);
    memcpy(&zero_h, p + 2, 2);
    float scale = qsf_fp16_to_fp32(scale_h);
    float zero  = qsf_fp16_to_fp32(zero_h);
    p += 4;

    for (int i = 0; i < count; i++) {
        int byte_idx = i / 2;
        uint8_t q;
        if (i % 2 == 0) {
            q = p[byte_idx] & 0x0F;
        } else {
            q = (p[byte_idx] >> 4) & 0x0F;
        }
        out[i] = scale * (float)q + zero;
    }
}

/* ── 4-bit symmetric dequantization ──────────────────────────────── */
void qsf_dequant_block_4bit_sym(const void* block_data, float* out, int count) {
    const uint8_t* p = (const uint8_t*)block_data;
    uint16_t scale_h;
    memcpy(&scale_h, p, 2);
    float scale = qsf_fp16_to_fp32(scale_h);
    p += 4;  /* skip scale + 2 bytes reserved */

    for (int i = 0; i < count; i++) {
        int byte_idx = i / 2;
        int8_t q;
        if (i % 2 == 0) {
            q = (int8_t)(p[byte_idx] & 0x0F);
        } else {
            q = (int8_t)((p[byte_idx] >> 4) & 0x0F);
        }
        /* Symmetric: signed range [-8, 7] */
        if (q >= 8) q -= 16;
        out[i] = scale * (float)q;
    }
}

/* ── Quantization (for KV cache and conversion tool) ─────────────── */
void qsf_quant_block_2bit(const float* values, void* out, int count) {
    /* Find min and max */
    float vmin = values[0], vmax = values[0];
    for (int i = 1; i < count; i++) {
        if (values[i] < vmin) vmin = values[i];
        if (values[i] > vmax) vmax = values[i];
    }

    float range = vmax - vmin;
    float scale = (range > 1e-30f) ? range / 3.0f : 0.0f;  /* 2-bit: 4 levels */
    float inv_scale = (scale > 1e-30f) ? 1.0f / scale : 0.0f;

    uint8_t* p = (uint8_t*)out;
    uint16_t scale_h = qsf_fp32_to_fp16(scale);
    uint16_t zero_h  = qsf_fp32_to_fp16(vmin);
    memcpy(p, &scale_h, 2);
    memcpy(p + 2, &zero_h, 2);
    p += 4;

    memset(p, 0, (count * 2 + 7) / 8);
    for (int i = 0; i < count; i++) {
        float normalized = (values[i] - vmin) * inv_scale;
        int q = (int)(normalized + 0.5f);
        if (q < 0) q = 0;
        if (q > 3) q = 3;
        int byte_idx = i / 4;
        int bit_off  = (i % 4) * 2;
        p[byte_idx] |= (uint8_t)(q << bit_off);
    }
}

void qsf_quant_block_3bit(const float* values, void* out, int count) {
    /* Find min and max */
    float vmin = values[0], vmax = values[0];
    for (int i = 1; i < count; i++) {
        if (values[i] < vmin) vmin = values[i];
        if (values[i] > vmax) vmax = values[i];
    }

    float range = vmax - vmin;
    float scale = (range > 1e-30f) ? range / 7.0f : 0.0f;  /* 3-bit: 8 levels */
    float inv_scale = (scale > 1e-30f) ? 1.0f / scale : 0.0f;

    uint8_t* p = (uint8_t*)out;
    uint16_t scale_h = qsf_fp32_to_fp16(scale);
    uint16_t zero_h  = qsf_fp32_to_fp16(vmin);
    memcpy(p, &scale_h, 2);
    memcpy(p + 2, &zero_h, 2);
    p += 4;

    /* Pack 3-bit values into bit stream */
    int total_bits = count * 3;
    int total_bytes = (total_bits + 7) / 8;
    memset(p, 0, total_bytes);

    int bit_pos = 0;
    for (int i = 0; i < count; i++) {
        float normalized = (values[i] - vmin) * inv_scale;
        int q = (int)(normalized + 0.5f);
        if (q < 0) q = 0;
        if (q > 7) q = 7;

        int byte_idx = bit_pos / 8;
        int bit_off  = bit_pos % 8;

        /* Write 3 bits; may span two bytes */
        p[byte_idx] |= (uint8_t)((q & 0x07) << bit_off);
        if (bit_off > 5) {
            /* Bits overflow into next byte */
            p[byte_idx + 1] |= (uint8_t)((q & 0x07) >> (8 - bit_off));
        }
        bit_pos += 3;
    }
}

void qsf_quant_block_4bit(const float* values, void* out, int count) {
    float vmin = values[0], vmax = values[0];
    for (int i = 1; i < count; i++) {
        if (values[i] < vmin) vmin = values[i];
        if (values[i] > vmax) vmax = values[i];
    }

    float range = vmax - vmin;
    float scale = (range > 1e-30f) ? range / 15.0f : 0.0f;
    float inv_scale = (scale > 1e-30f) ? 1.0f / scale : 0.0f;

    uint8_t* p = (uint8_t*)out;
    uint16_t scale_h = qsf_fp32_to_fp16(scale);
    uint16_t zero_h  = qsf_fp32_to_fp16(vmin);
    memcpy(p, &scale_h, 2);
    memcpy(p + 2, &zero_h, 2);
    p += 4;

    memset(p, 0, (count * 4 + 7) / 8);
    for (int i = 0; i < count; i++) {
        float normalized = (values[i] - vmin) * inv_scale;
        int q = (int)(normalized + 0.5f);
        if (q < 0) q = 0;
        if (q > 15) q = 15;
        int byte_idx = i / 2;
        if (i % 2 == 0) {
            p[byte_idx] |= (uint8_t)(q & 0x0F);
        } else {
            p[byte_idx] |= (uint8_t)((q & 0x0F) << 4);
        }
    }
}
