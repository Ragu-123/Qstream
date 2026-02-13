/*
 * QStream - engine.h
 * Inference engine: manages model, memory, KV cache, and generation.
 */
#ifndef QSF_ENGINE_H
#define QSF_ENGINE_H

#include "types.h"
#include "error.h"
#include "format.h"
#include "tokenizer.h"
#include "arena.h"
#include "kernels.h"
#include "sampling.h"
#include "platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Token callback: called after each generated token. Return 0 to continue. */
typedef int (*qsf_token_callback)(uint32_t token_id, const char* text, void* userdata);

/* Engine configuration */
typedef struct {
    size_t   ram_budget;       /* bytes, 0 = auto-detect */
    int      max_seq_len;      /* 0 = use model default */
    int      num_threads;      /* 0 = auto-detect */
    int      allow_mmap;       /* 1 = yes (default) */
    int      verbose;          /* 0 = quiet, 1 = info, 2 = debug */
    int      check_nan;        /* 0 = off (default), 1 = per-layer NaN check */
} QSFEngineConfig;

/* KV cache for one layer, one head */
typedef struct {
    void*   k_cache;           /* quantized key cache [max_seq × head_dim] */
    void*   v_cache;           /* quantized value cache [max_seq × head_dim] */
    float*  k_scales;          /* scale per position */
    float*  v_scales;          /* scale per position */
} QSFKVHead;

/* Full KV cache */
typedef struct {
    QSFKVHead* heads;          /* [num_layers * num_kv_heads] */
    int        num_layers;
    int        num_kv_heads;
    int        head_dim;
    int        max_seq;
    int        kv_quant_bits;
    int        current_seq_len; /* how many positions are filled */
} QSFKVCache;

/* Memory budget allocation result */
typedef struct {
    size_t   total_budget;
    size_t   layer_buf_size;
    size_t   activation_size;
    size_t   scratch_size;
    size_t   kv_cache_size;
    size_t   embedding_cache_size;
    int      kv_max_seq;
    int      kv_quant_bits;
    int      enable_prefetch;
    int      max_prompt_chunk;
} QSFBudget;

/* The inference engine */
typedef struct {
    QSFModel         model;
    QSFTokenizer     tokenizer;
    QSFKernelTable   kernels;
    QSFPlatformInfo  platform;
    Arena*           arena;
    QSFBudget        budget;
    QSFKVCache       kv_cache;

    /* Working buffers (allocated from arena) */
    float*   activation;       /* [hidden_dim] current activation */
    float*   residual;         /* [hidden_dim] residual stream */
    float*   scratch;          /* [max(hidden_dim, intermediate_dim)] */
    float*   scratch2;         /* second scratch for gated FFN */
    float*   attn_scores;     /* [max_seq] per-head attention scores */
    float*   logits;           /* [vocab_size] output logits */
    float*   q_buf;            /* [num_heads * head_dim] */
    float*   k_buf;            /* [num_kv_heads * head_dim] */
    float*   v_buf;            /* [num_kv_heads * head_dim] */
    float*   attn_out;         /* [hidden_dim] attention output */

    /* Layer buffers */
    uint8_t* layer_buf;        /* decompressed layer data */
    uint8_t* layer_buf_compressed; /* compressed layer data (if not mmap) */

    /* Norm weights (kept in memory for all layers) */
    float*   attn_norm_weights; /* [num_layers * hidden_dim] */
    float*   attn_norm_biases;  /* [num_layers * hidden_dim] or NULL */
    float*   ffn_norm_weights;  /* [num_layers * hidden_dim] */
    float*   ffn_norm_biases;   /* [num_layers * hidden_dim] or NULL */
    float*   final_norm_weight; /* [hidden_dim] */
    float*   final_norm_bias;   /* [hidden_dim] or NULL */

    /* Position embedding table (GPT-2 only) */
    float*   pos_embed;        /* [max_seq * hidden_dim] or NULL */

    /* Output head */
    void*    output_head;      /* quantized weight or NULL if tied */
    int      output_head_quant;

    /* State */
    volatile int interrupted;
    int      loaded;
    int      check_nan;        /* NaN detection enabled */
    int      verbose;          /* verbosity level */
} QSFEngine;

/* Initialize config with defaults */
void     qsf_engine_config_default(QSFEngineConfig* cfg);

/* Create engine: loads model, allocates memory, initializes everything */
QSFError qsf_engine_create(QSFEngine* engine, const char* model_path,
                            const QSFEngineConfig* cfg);

/* Compute memory budget */
QSFError qsf_budget_compute(QSFBudget* budget, const QSFHeader* h,
                             const QSFLayerIndex* li, size_t ram_budget);

/* Run a single forward pass for one token at given position.
   Returns logits in engine->logits. */
QSFError qsf_forward(QSFEngine* engine, uint32_t token_id, int position);

/* Generate tokens from a prompt string.
   Calls callback for each generated token. */
QSFError qsf_engine_generate(QSFEngine* engine,
                              const char* prompt,
                              int max_new_tokens,
                              const QSFSamplingConfig* sampling,
                              qsf_token_callback callback,
                              void* userdata);

/* Request interrupt (safe to call from signal handler) */
void     qsf_engine_interrupt(QSFEngine* engine);

/* Free all engine resources */
void     qsf_engine_free(QSFEngine* engine);

#ifdef __cplusplus
}
#endif
#endif /* QSF_ENGINE_H */
