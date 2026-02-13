/*
 * QStream - CPU-Optimized LLM Inference Engine
 * types.h - Core type definitions, enums, on-disk structures
 *
 * All on-disk structures are packed, little-endian, use fixed-width types.
 */
#ifndef QSF_TYPES_H
#define QSF_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Magic & Version ─────────────────────────────────────────────── */
#define QSF_MAGIC         0x51534631u  /* "QSF1" */
#define QSF_VERSION       1u
#define QSF_HEADER_SIZE   256u
#define QSF_ALIGNMENT     64u
#define QSF_BLOCK_SIZE    64u          /* default quantization block */
#define QSF_MAX_LAYERS    1000u
#define QSF_NO_TOKEN      0xFFFFFFFFu  /* sentinel for absent special tokens */

/* ── Architecture Enum ───────────────────────────────────────────── */
typedef enum {
    QSF_ARCH_GPT2       = 0,
    QSF_ARCH_LLAMA      = 1,
    QSF_ARCH_MISTRAL    = 2,
    QSF_ARCH_PHI        = 3,
    QSF_ARCH_GPTJ       = 4,
    QSF_ARCH_QWEN       = 5,
    QSF_ARCH_GEMMA      = 6,
    QSF_ARCH_STABLELM   = 7,
    QSF_ARCH_MIXTRAL    = 8,
    QSF_ARCH_DEEPSEEK   = 9,
    QSF_ARCH_GPT_OSS    = 10,
    QSF_ARCH_CUSTOM     = 128,
} QSFArch;

/* ── Quantization Type ───────────────────────────────────────────── */
typedef enum {
    QSF_QUANT_2BIT_ASYM  = 0,
    QSF_QUANT_2BIT_SYM   = 1,
    QSF_QUANT_3BIT_ASYM  = 2,
    QSF_QUANT_4BIT_ASYM  = 3,
    QSF_QUANT_4BIT_SYM   = 4,
    QSF_QUANT_TERNARY    = 5,   /* 1.58-bit BitNet */
    QSF_QUANT_8BIT       = 6,
    QSF_QUANT_FP16       = 7,
    QSF_QUANT_MIXED      = 8,
    QSF_QUANT_USE_DEFAULT = 0xFF,
} QSFQuantType;

/* ── Activation Function ─────────────────────────────────────────── */
typedef enum {
    QSF_ACT_GELU         = 0,
    QSF_ACT_GELU_APPROX  = 1,
    QSF_ACT_SILU         = 2,   /* Swish, used by LLaMA */
    QSF_ACT_RELU         = 3,
    QSF_ACT_RELU_SQ      = 4,
    QSF_ACT_GEGLU        = 5,
} QSFActivation;

/* ── Normalization Type ──────────────────────────────────────────── */
typedef enum {
    QSF_NORM_LAYERNORM_PRE   = 0,
    QSF_NORM_LAYERNORM_POST  = 1,
    QSF_NORM_RMSNORM_PRE     = 2,
    QSF_NORM_RMSNORM_POST    = 3,
} QSFNormType;

/* ── Position Encoding ───────────────────────────────────────────── */
typedef enum {
    QSF_POS_LEARNED   = 0,
    QSF_POS_ROPE      = 1,
    QSF_POS_ALIBI     = 2,
    QSF_POS_NONE      = 3,
    QSF_POS_RELATIVE  = 4,
} QSFPosEncoding;

/* ── Compression Type ────────────────────────────────────────────── */
typedef enum {
    QSF_COMPRESS_NONE  = 0,
    QSF_COMPRESS_LZ4   = 1,
    QSF_COMPRESS_LZ4HC = 2,
    QSF_COMPRESS_ZSTD  = 3,
} QSFCompressionType;

/* ── FFN Type ────────────────────────────────────────────────────── */
typedef enum {
    QSF_FFN_STANDARD  = 0,   /* up → act → down */
    QSF_FFN_GATED     = 1,   /* gate ⊙ up → down (LLaMA) */
    QSF_FFN_PARALLEL  = 2,   /* GPT-J parallel attn+FFN */
    QSF_FFN_MOE       = 3,   /* Mixture-of-Experts sparse FFN */
} QSFFFNType;

/* ── Attention Type ──────────────────────────────────────────────── */
typedef enum {
    QSF_ATTN_SEPARATE   = 0,
    QSF_ATTN_FUSED_QKV  = 1,
    QSF_ATTN_FUSED_QKV_SEPARATE_O = 2,
} QSFAttnType;

/* ── RoPE Scaling Type ───────────────────────────────────────────── */
typedef enum {
    QSF_ROPE_NONE      = 0,
    QSF_ROPE_LINEAR    = 1,
    QSF_ROPE_NTK       = 2,
    QSF_ROPE_YARN      = 3,
    QSF_ROPE_DYN_NTK   = 4,
} QSFRopeScaling;

/* ── Tensor Type ─────────────────────────────────────────────────── */
typedef enum {
    QSF_TENSOR_ATTN_Q       = 0,
    QSF_TENSOR_ATTN_K       = 1,
    QSF_TENSOR_ATTN_V       = 2,
    QSF_TENSOR_ATTN_O       = 3,
    QSF_TENSOR_FFN_GATE     = 4,
    QSF_TENSOR_FFN_UP       = 5,
    QSF_TENSOR_FFN_DOWN     = 6,
    QSF_TENSOR_ATTN_NORM_W  = 7,
    QSF_TENSOR_ATTN_NORM_B  = 8,
    QSF_TENSOR_FFN_NORM_W   = 9,
    QSF_TENSOR_FFN_NORM_B   = 10,
    QSF_TENSOR_ATTN_Q_BIAS  = 11,
    QSF_TENSOR_ATTN_K_BIAS  = 12,
    QSF_TENSOR_ATTN_V_BIAS  = 13,
    QSF_TENSOR_ATTN_O_BIAS  = 14,
    QSF_TENSOR_FFN_GATE_BIAS = 15,
    QSF_TENSOR_FFN_UP_BIAS  = 16,
    QSF_TENSOR_FFN_DOWN_BIAS = 17,
    QSF_TENSOR_FUSED_QKV    = 18,
    QSF_TENSOR_FUSED_QKV_BIAS = 19,
    QSF_TENSOR_POS_EMBED    = 20,
    /* MoE tensors — IDs encode expert index (stride 6):
     *   GATE(e) = GATE_0 + e*6
     *   GATE_BIAS(e) = GATE_BIAS_0 + e*6
     *   ...
     */
    QSF_TENSOR_MOE_ROUTER      = 21,
    QSF_TENSOR_MOE_ROUTER_BIAS = 22,
    QSF_TENSOR_MOE_GATE_0      = 23,
    QSF_TENSOR_MOE_GATE_BIAS_0 = 24,
    QSF_TENSOR_MOE_UP_0        = 25,
    QSF_TENSOR_MOE_UP_BIAS_0   = 26,
    QSF_TENSOR_MOE_DOWN_0      = 27,
    QSF_TENSOR_MOE_DOWN_BIAS_0 = 28,
} QSFTensorType;

/* Helper macros for MoE expert tensor IDs */
#define QSF_MOE_STRIDE 6
#define QSF_MOE_GATE(e)       (QSF_TENSOR_MOE_GATE_0      + (e) * QSF_MOE_STRIDE)
#define QSF_MOE_GATE_BIAS(e)  (QSF_TENSOR_MOE_GATE_BIAS_0 + (e) * QSF_MOE_STRIDE)
#define QSF_MOE_UP(e)         (QSF_TENSOR_MOE_UP_0        + (e) * QSF_MOE_STRIDE)
#define QSF_MOE_UP_BIAS(e)    (QSF_TENSOR_MOE_UP_BIAS_0   + (e) * QSF_MOE_STRIDE)
#define QSF_MOE_DOWN(e)       (QSF_TENSOR_MOE_DOWN_0      + (e) * QSF_MOE_STRIDE)
#define QSF_MOE_DOWN_BIAS(e)  (QSF_TENSOR_MOE_DOWN_BIAS_0 + (e) * QSF_MOE_STRIDE)

/* ── Tokenizer Type ──────────────────────────────────────────────── */
typedef enum {
    QSF_TOK_BPE           = 0,
    QSF_TOK_SP_BPE        = 1,
    QSF_TOK_SP_UNIGRAM    = 2,
    QSF_TOK_WORDPIECE     = 3,
} QSFTokenizerType;

/*
 * ══════════════════════════════════════════════════════════════════
 *  ON-DISK STRUCTURES  (packed, little-endian)
 * ══════════════════════════════════════════════════════════════════
 */
#ifdef _MSC_VER
  #pragma pack(push, 1)
  #define QSF_PACKED
#else
  #define QSF_PACKED __attribute__((packed))
#endif

/* File header — 256 bytes */
typedef struct QSF_PACKED {
    uint32_t magic;                 /*   0: "QSF1" */
    uint32_t version;               /*   4 */
    uint32_t header_size;           /*   8 */
    uint32_t arch;                  /*  12 */
    uint32_t num_layers;            /*  16 */
    uint32_t hidden_dim;            /*  20 */
    uint32_t num_heads;             /*  24 */
    uint32_t num_kv_heads;          /*  28 */
    uint32_t vocab_size;            /*  32 */
    uint32_t max_seq_len;           /*  36 */
    uint32_t intermediate_dim;      /*  40 */
    uint32_t head_dim;              /*  44 */
    uint8_t  quant_type;            /*  48 */
    uint8_t  activation;            /*  49 */
    uint8_t  norm_type;             /*  50 */
    uint8_t  pos_enc_type;          /*  51 */
    float    rope_theta;            /*  52 */
    float    norm_epsilon;          /*  56 */
    uint32_t block_size;            /*  60 */

    /* Section offsets (64-byte aligned) */
    uint64_t layer_index_offset;    /*  64 */
    uint64_t embedding_offset;      /*  72 */
    uint64_t final_offset;          /*  80 */
    uint64_t tokenizer_offset;      /*  88 */
    uint64_t extended_config_offset;/*  96 */
    uint64_t calibration_offset;    /* 104 */
    uint64_t importance_offset;     /* 112 */

    /* Special token IDs */
    uint32_t bos_token_id;          /* 120 */
    uint32_t eos_token_id;          /* 124 */
    uint32_t pad_token_id;          /* 128 */
    uint32_t unk_token_id;          /* 132 */

    /* Model metadata */
    uint32_t num_params_millions;   /* 136 */
    uint8_t  has_bias;              /* 140: bitfield */
    uint8_t  ffn_type;              /* 141 */
    uint8_t  attn_type;             /* 142 */
    uint8_t  weight_tying;          /* 143 */
    uint32_t rope_scaling_type;     /* 144 */
    float    rope_scaling_factor;   /* 148 */
    uint32_t sliding_window;        /* 152 */
    uint32_t tie_word_embeddings;   /* 156 */

    /* Integrity */
    uint64_t total_file_size;       /* 160 */
    uint32_t header_crc32;          /* 168 */
    uint8_t  endian_marker;         /* 172: 0x01 = LE */

    /* MoE configuration (carved from reserved, backward compat:
     * num_experts==0 means dense, which was the old default) */
    uint8_t  num_experts;           /* 173: 0/1 = dense, >1 = MoE */
    uint8_t  num_active_experts;    /* 174: top-K (e.g. 2) */
    uint8_t  moe_norm_topk;         /* 175: 1 = renormalize router weights */
    uint8_t  reserved_moe;          /* 176: padding */
    uint32_t expert_intermediate_dim; /* 177: per-expert FFN dim */
    uint8_t  reserved[75];          /* 181-255 */
} QSFHeader;

/* Layer index entry — 48 bytes */
typedef struct QSF_PACKED {
    uint64_t offset;
    uint32_t compressed_size;
    uint32_t decompressed_size;
    uint8_t  quant_type;
    uint8_t  compression_type;
    uint16_t num_tensors;
    uint32_t crc32_compressed;
    uint32_t crc32_decompressed;
    float    importance_score;
    uint32_t weight_bytes;
    uint32_t layer_ffn_dim;
    uint8_t  reserved[8];
} QSFLayerIndex;

/* Tensor header — 24 bytes */
typedef struct QSF_PACKED {
    uint16_t tensor_type;
    uint8_t  quant_type;            /* 0xFF = use layer default */
    uint8_t  data_layout;           /* 0=row-major, 1=col-major, 2=transposed */
    uint32_t rows;
    uint32_t cols;
    uint32_t data_size;             /* bytes, not including header */
    uint32_t num_outliers;
    uint32_t reserved;
} QSFTensorHeader;

/* Embedding section header — 32 bytes */
typedef struct QSF_PACKED {
    uint32_t quant_type;
    uint32_t compressed_size;
    uint32_t num_vectors;
    uint32_t embedding_dim;
    uint8_t  compression_type;
    uint8_t  reserved1[3];
    uint32_t crc32;
    uint32_t num_chunks;
    uint32_t chunk_size;
} QSFEmbeddingHeader;

/* Tokenizer header — 32 bytes */
typedef struct QSF_PACKED {
    uint32_t tokenizer_type;
    uint32_t vocab_size;
    uint32_t num_merges;
    uint32_t num_added_tokens;
    uint32_t vocab_data_size;
    uint32_t merge_data_size;
    uint32_t added_tokens_data_size;
    uint32_t flags;
} QSFTokenizerHeader;

/* Final section header — 16 bytes */
typedef struct QSF_PACKED {
    uint32_t output_head_type;     /* 0=separate, 1=tied, 2=separate+bias */
    uint32_t final_norm_size;
    uint32_t output_head_size;
    uint32_t crc32;
} QSFFinalHeader;

/* Outlier entry — 6 bytes */
typedef struct QSF_PACKED {
    uint32_t flat_index;
    uint16_t fp16_value;
} QSFOutlierEntry;

/* BPE merge rule — 12 bytes */
typedef struct QSF_PACKED {
    uint32_t token_a;
    uint32_t token_b;
    uint32_t merged;
} QSFMergeRule;

#ifdef _MSC_VER
  #pragma pack(pop)
#endif

/* ── Compile-time checks ─────────────────────────────────────────── */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
  _Static_assert(sizeof(QSFHeader)         == 256, "QSFHeader must be 256 bytes");
  _Static_assert(sizeof(QSFLayerIndex)     == 48,  "QSFLayerIndex must be 48 bytes");
  _Static_assert(sizeof(QSFTensorHeader)   == 24,  "QSFTensorHeader must be 24 bytes");
  _Static_assert(sizeof(QSFEmbeddingHeader)== 32,  "QSFEmbeddingHeader must be 32 bytes");
  _Static_assert(sizeof(QSFTokenizerHeader)== 32,  "QSFTokenizerHeader must be 32 bytes");
  _Static_assert(sizeof(QSFFinalHeader)    == 16,  "QSFFinalHeader must be 16 bytes");
  _Static_assert(sizeof(float)             == 4,   "float must be 32 bits");
#endif

/* ── Utility macros ──────────────────────────────────────────────── */
#define QSF_ALIGN_UP(x, a)  (((x) + (a) - 1) & ~((a) - 1))
#define QSF_MIN(a, b)       ((a) < (b) ? (a) : (b))
#define QSF_MAX(a, b)       ((a) > (b) ? (a) : (b))

#ifdef __cplusplus
}
#endif
#endif /* QSF_TYPES_H */
