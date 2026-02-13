/*
 * QStream - engine.c
 * The inference engine: budget computation, forward pass, generation loop.
 * This is the heart of QStream — the transformer inference pipeline.
 */
#include "qsf/engine.h"
#include "qsf/quant.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Config defaults ─────────────────────────────────────────────── */
void qsf_engine_config_default(QSFEngineConfig* cfg) {
    cfg->ram_budget  = 0;    /* auto-detect */
    cfg->max_seq_len = 0;    /* model default */
    cfg->num_threads = 0;    /* auto */
    cfg->allow_mmap  = 1;
    cfg->verbose     = 0;
    cfg->check_nan   = 0;
}

/* ── Budget computation ──────────────────────────────────────────── */
QSFError qsf_budget_compute(QSFBudget* budget, const QSFHeader* h,
                             const QSFLayerIndex* li, size_t ram_budget) {
    memset(budget, 0, sizeof(*budget));
    budget->total_budget = ram_budget;

    uint32_t hd = h->hidden_dim;
    uint32_t id = h->intermediate_dim;
    if (id == 0) id = hd * 4;
    uint32_t nl = h->num_layers;
    uint32_t nh = h->num_heads;
    uint32_t nkv = h->num_kv_heads;
    if (nkv == 0) nkv = nh;
    uint32_t head_d = h->head_dim;
    if (head_d == 0) head_d = hd / nh;
    int gated = (h->ffn_type == QSF_FFN_GATED);

    /* Priority 1: minimum required */
    size_t norm_cost = nl * hd * 2 * 2;  /* attn + ffn norms, fp16 */
    size_t final_norm = hd * 2;
    size_t activation = hd * 4;
    size_t scratch = (id > hd ? id : hd) * 4;
    if (gated) scratch += id * 4;  /* second scratch for gate */
    size_t qkv_buf = (nh + 2 * nkv) * head_d * 4;
    size_t attn_out_buf = hd * 4;
    size_t logits_buf = h->vocab_size * 4;

    /* Max layer buffer */
    uint32_t max_comp = 0, max_decomp = 0;
    for (uint32_t i = 0; i < nl; i++) {
        if (li[i].compressed_size > max_comp) max_comp = li[i].compressed_size;
        if (li[i].decompressed_size > max_decomp) max_decomp = li[i].decompressed_size;
    }
    budget->layer_buf_size = max_comp + max_decomp + 4096; /* padding */

    size_t p1 = norm_cost + final_norm + activation + scratch + qkv_buf +
                attn_out_buf + logits_buf + budget->layer_buf_size;

    if (p1 > ram_budget) {
        qsf_set_error(QSF_ERR_BUDGET_EXCEEDED,
                       "minimum allocations exceed budget");
        return QSF_ERR_BUDGET_EXCEEDED;
    }

    budget->activation_size = activation;
    budget->scratch_size = scratch;

    /* Priority 2: KV cache */
    size_t remaining = ram_budget - p1;
    size_t kv_per_pos_per_layer = 2 * nkv * head_d;  /* 4-bit: /2 */

    /* Try different configs */
    int kv_bits_options[] = {4, 4, 4, 2, 2};
    int kv_seq_options[]  = {2048, 1024, 512, 1024, 256};
    int num_configs = 5;
    int chosen = -1;

    for (int c = 0; c < num_configs; c++) {
        int win = kv_seq_options[c];
        if (h->max_seq_len > 0 && win > (int)h->max_seq_len)
            win = (int)h->max_seq_len;

        size_t bytes_per_elem = (kv_bits_options[c] == 4) ? 1 : 1; /* packed */
        size_t kv_cost = (size_t)win * nl * kv_per_pos_per_layer * bytes_per_elem / 2;
        /* Add scales: one float per position per head per K/V */
        kv_cost += (size_t)win * nl * nkv * 2 * 4;
        /* Attention scores buffer */
        size_t attn_score = nh * win * 4;
        kv_cost += attn_score;

        if (kv_cost < remaining * 85 / 100) {
            budget->kv_cache_size = kv_cost;
            budget->kv_max_seq    = win;
            budget->kv_quant_bits = kv_bits_options[c];
            chosen = c;
            break;
        }
    }

    if (chosen < 0) {
        /* Absolute minimum: 128 sequence length, 4-bit */
        budget->kv_max_seq = 128;
        budget->kv_quant_bits = 4;
        budget->kv_cache_size = (size_t)128 * nl * kv_per_pos_per_layer / 2 +
                                (size_t)128 * nl * nkv * 2 * 4 +
                                nh * 128 * 4;
    }

    remaining = ram_budget - p1 - budget->kv_cache_size;

    /* Priority 3: double-buffer for layer prefetch */
    if (remaining >= budget->layer_buf_size) {
        budget->enable_prefetch = 1;
        remaining -= budget->layer_buf_size;
    }

    /* Priority 3: embedding cache */
    size_t emb_entry = hd * 2;
    if (remaining >= 500 * emb_entry) {
        budget->embedding_cache_size = 500 * emb_entry;
        remaining -= budget->embedding_cache_size;
    }

    /* Prompt batch size */
    budget->max_prompt_chunk = (int)(remaining / (hd * 4));
    if (budget->max_prompt_chunk < 1) budget->max_prompt_chunk = 1;
    if (budget->max_prompt_chunk > 128) budget->max_prompt_chunk = 128;

    return QSF_OK;
}

/* ── Helper: get matvec function for a quant type ────────────────── */
static qsf_matvec_fn get_matvec_fn(const QSFKernelTable* kt, uint8_t qt) {
    switch (qt) {
        case QSF_QUANT_2BIT_ASYM:
        case QSF_QUANT_2BIT_SYM:  return kt->matvec_2bit;
        case QSF_QUANT_3BIT_ASYM: return kt->matvec_3bit;
        case QSF_QUANT_4BIT_SYM:  return kt->matvec_4bit_sym;
        case QSF_QUANT_4BIT_ASYM:
        default:                  return kt->matvec_4bit;
    }
}

/* ── Helper: find tensor in decompressed layer data ──────────────── */
static const void* find_tensor(const uint8_t* layer_data, int num_tensors,
                                uint16_t target_type, QSFTensorHeader* out_hdr) {
    const uint8_t* p = layer_data;
    for (int t = 0; t < num_tensors; t++) {
        QSFTensorHeader th;
        memcpy(&th, p, sizeof(th));
        p += sizeof(QSFTensorHeader);
        if (th.tensor_type == target_type) {
            if (out_hdr) *out_hdr = th;
            return p;
        }
        p += th.data_size;
        /* Skip outlier table */
        if (th.num_outliers > 0) {
            p += th.num_outliers * sizeof(QSFOutlierEntry);
        }
    }
    if (out_hdr) memset(out_hdr, 0, sizeof(*out_hdr));
    return NULL;
}

/* ── Helper: RoPE application ────────────────────────────────────── */
static void apply_rope(float* q, float* k, int head_dim, int position,
                       float theta, int num_heads, int num_kv_heads) {
    /* Apply RoPE to each head */
    for (int h = 0; h < num_heads; h++) {
        float* qh = q + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / (float)head_dim);
            double angle = (double)position * (double)freq;
            angle = fmod(angle, 2.0 * M_PI);
            float cos_a = cosf((float)angle);
            float sin_a = sinf((float)angle);
            float q0 = qh[i], q1 = qh[i + 1];
            qh[i]   = q0 * cos_a - q1 * sin_a;
            qh[i+1] = q0 * sin_a + q1 * cos_a;
        }
    }
    for (int h = 0; h < num_kv_heads; h++) {
        float* kh = k + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / (float)head_dim);
            double angle = (double)position * (double)freq;
            angle = fmod(angle, 2.0 * M_PI);
            float cos_a = cosf((float)angle);
            float sin_a = sinf((float)angle);
            float k0 = kh[i], k1 = kh[i + 1];
            kh[i]   = k0 * cos_a - k1 * sin_a;
            kh[i+1] = k0 * sin_a + k1 * cos_a;
        }
    }
}

/* ── Helper: KV cache store (simple float storage for now) ───────── */
static void kv_cache_store(QSFKVCache* kv, int layer, int position,
                           const float* k_vec, const float* v_vec) {
    int hd = kv->head_dim;
    int nkv = kv->num_kv_heads;

    for (int h = 0; h < nkv; h++) {
        int head_idx = layer * nkv + h;
        QSFKVHead* head = &kv->heads[head_idx];

        /* Store K */
        float* k_dst = (float*)head->k_cache + position * hd;
        memcpy(k_dst, k_vec + h * hd, hd * sizeof(float));
        head->k_scales[position] = 1.0f;

        /* Store V */
        float* v_dst = (float*)head->v_cache + position * hd;
        memcpy(v_dst, v_vec + h * hd, hd * sizeof(float));
        head->v_scales[position] = 1.0f;
    }
}

/* ── Allocate KV cache ───────────────────────────────────────────── */
static QSFError alloc_kv_cache(QSFKVCache* kv, Arena* arena,
                                int num_layers, int num_kv_heads,
                                int head_dim, int max_seq) {
    kv->num_layers   = num_layers;
    kv->num_kv_heads = num_kv_heads;
    kv->head_dim     = head_dim;
    kv->max_seq      = max_seq;
    kv->current_seq_len = 0;

    int total_heads = num_layers * num_kv_heads;
    kv->heads = (QSFKVHead*)arena_alloc(arena, total_heads * sizeof(QSFKVHead),
                                         64, "kv_heads");
    if (!kv->heads) return QSF_ERR_OUT_OF_MEMORY;

    for (int i = 0; i < total_heads; i++) {
        kv->heads[i].k_cache = arena_alloc(arena, max_seq * head_dim * sizeof(float),
                                            64, "k_cache");
        kv->heads[i].v_cache = arena_alloc(arena, max_seq * head_dim * sizeof(float),
                                            64, "v_cache");
        kv->heads[i].k_scales = (float*)arena_alloc(arena, max_seq * sizeof(float),
                                                      64, "k_scales");
        kv->heads[i].v_scales = (float*)arena_alloc(arena, max_seq * sizeof(float),
                                                      64, "v_scales");
        if (!kv->heads[i].k_cache || !kv->heads[i].v_cache ||
            !kv->heads[i].k_scales || !kv->heads[i].v_scales)
            return QSF_ERR_OUT_OF_MEMORY;
    }

    return QSF_OK;
}

/* ── Load norm weights from QSF file ─────────────────────────────── */
static QSFError load_norms(QSFEngine* engine) {
    const QSFHeader* h = &engine->model.header;
    uint32_t nl = h->num_layers;
    uint32_t hd = h->hidden_dim;

    /* Allocate norm weight arrays */
    engine->attn_norm_weights = (float*)arena_alloc(engine->arena,
        nl * hd * sizeof(float), 64, "attn_norm_w");
    engine->ffn_norm_weights = (float*)arena_alloc(engine->arena,
        nl * hd * sizeof(float), 64, "ffn_norm_w");
    engine->final_norm_weight = (float*)arena_alloc(engine->arena,
        hd * sizeof(float), 64, "final_norm_w");

    if (!engine->attn_norm_weights || !engine->ffn_norm_weights ||
        !engine->final_norm_weight)
        return QSF_ERR_OUT_OF_MEMORY;

    if (engine->model.has_bias_norm) {
        engine->attn_norm_biases = (float*)arena_alloc(engine->arena,
            nl * hd * sizeof(float), 64, "attn_norm_b");
        engine->ffn_norm_biases = (float*)arena_alloc(engine->arena,
            nl * hd * sizeof(float), 64, "ffn_norm_b");
        engine->final_norm_bias = (float*)arena_alloc(engine->arena,
            hd * sizeof(float), 64, "final_norm_b");
    }

    /* Load norms from each layer's data */
    for (uint32_t layer = 0; layer < nl; layer++) {
        QSFLayerIndex* li = &engine->model.layer_index[layer];
        size_t data_size = (li->compression_type == 0) ?
                           li->compressed_size : li->decompressed_size;

        const uint8_t* layer_data;
        if (li->compression_type == 0) {
            layer_data = (const uint8_t*)file_access_get(
                &engine->model.file, (size_t)li->offset, li->compressed_size);
        } else {
            /* For compressed layers, we need to decompress to get norms.
               Norms are tiny (fp16), stored as tensors within layer data.
               For now, load the full layer and extract. */
            layer_data = (const uint8_t*)file_access_get(
                &engine->model.file, (size_t)li->offset, li->compressed_size);
            /* TODO: decompress if compression is enabled */
        }
        if (!layer_data) return QSF_ERR_IO_FAILURE;

        /* Find attention norm tensor */
        QSFTensorHeader th;
        const void* norm_data = find_tensor(layer_data, li->num_tensors,
                                             QSF_TENSOR_ATTN_NORM_W, &th);
        if (norm_data && th.data_size >= hd * 2) {
            engine->kernels.fp16_to_fp32(
                (const uint16_t*)norm_data,
                engine->attn_norm_weights + layer * hd, hd);
        } else {
            /* Default: all ones */
            for (uint32_t i = 0; i < hd; i++)
                engine->attn_norm_weights[layer * hd + i] = 1.0f;
        }

        /* Find FFN norm tensor */
        norm_data = find_tensor(layer_data, li->num_tensors,
                                QSF_TENSOR_FFN_NORM_W, &th);
        if (norm_data && th.data_size >= hd * 2) {
            engine->kernels.fp16_to_fp32(
                (const uint16_t*)norm_data,
                engine->ffn_norm_weights + layer * hd, hd);
        } else {
            for (uint32_t i = 0; i < hd; i++)
                engine->ffn_norm_weights[layer * hd + i] = 1.0f;
        }
    }

    /* Load final norm from final section */
    uint64_t final_off = h->final_offset;
    if (final_off > 0 && final_off + sizeof(QSFFinalHeader) <= engine->model.file.file_size) {
        const void* final_data = file_access_get(&engine->model.file,
            (size_t)final_off, sizeof(QSFFinalHeader));
        if (final_data) {
            QSFFinalHeader fh;
            memcpy(&fh, final_data, sizeof(fh));

            const void* norm_bytes = file_access_get(&engine->model.file,
                (size_t)final_off + sizeof(QSFFinalHeader), hd * 2);
            if (norm_bytes) {
                engine->kernels.fp16_to_fp32(
                    (const uint16_t*)norm_bytes,
                    engine->final_norm_weight, hd);
            }
        }
    } else {
        /* Default */
        for (uint32_t i = 0; i < hd; i++)
            engine->final_norm_weight[i] = 1.0f;
    }

    return QSF_OK;
}

/* ── Create engine ───────────────────────────────────────────────── */
QSFError qsf_engine_create(QSFEngine* engine, const char* model_path,
                            const QSFEngineConfig* cfg) {
    memset(engine, 0, sizeof(*engine));

    /* Detect platform */
    qsf_detect_platform(&engine->platform);
    qsf_flush_denormals();

    /* Initialize kernels */
    qsf_kernels_init(&engine->kernels, &engine->platform);

    if (cfg->verbose >= 1) {
        fprintf(stderr, "[qstream] Platform: %s, cores=%d, RAM=%.0f MB\n",
                engine->platform.best_isa == QSF_ISA_AVX2 ? "AVX2" :
                engine->platform.best_isa == QSF_ISA_AVX512 ? "AVX-512" :
                engine->platform.best_isa == QSF_ISA_NEON ? "NEON" : "scalar",
                engine->platform.num_cores,
                (double)engine->platform.total_ram_bytes / (1024*1024));
    }

    /* Load model */
    QSFError err = qsf_model_load(&engine->model, model_path, cfg->allow_mmap);
    if (err != QSF_OK) return err;

    const QSFHeader* h = &engine->model.header;
    if (cfg->verbose >= 1) {
        fprintf(stderr, "[qstream] Model: arch=%u, layers=%u, hidden=%u, heads=%u, "
                "kv_heads=%u, vocab=%u\n",
                h->arch, h->num_layers, h->hidden_dim, h->num_heads,
                engine->model.num_kv_heads, h->vocab_size);
    }

    /* Compute budget */
    size_t budget_bytes = cfg->ram_budget;
    if (budget_bytes == 0) {
        uint64_t avail = engine->platform.available_ram_bytes;
        if (avail == 0) avail = 200ULL * 1024 * 1024;
        budget_bytes = (size_t)(avail / 2);
        if (budget_bytes > 200ULL * 1024 * 1024)
            budget_bytes = 200ULL * 1024 * 1024;
        if (budget_bytes < 64ULL * 1024 * 1024)
            budget_bytes = 64ULL * 1024 * 1024;
    }

    err = qsf_budget_compute(&engine->budget, h,
                              engine->model.layer_index, budget_bytes);
    if (err != QSF_OK) {
        qsf_model_free(&engine->model);
        return err;
    }

    if (cfg->verbose >= 1) {
        fprintf(stderr, "[qstream] Budget: %zu MB, KV cache: %d seq @ %d-bit\n",
                budget_bytes / (1024*1024),
                engine->budget.kv_max_seq, engine->budget.kv_quant_bits);
    }

    /* Create arena */
    engine->arena = arena_create(budget_bytes);
    if (!engine->arena) {
        qsf_model_free(&engine->model);
        qsf_set_error(QSF_ERR_OUT_OF_MEMORY, "arena create");
        return QSF_ERR_OUT_OF_MEMORY;
    }

    uint32_t hd = h->hidden_dim;
    uint32_t id = h->intermediate_dim;
    uint32_t bs = h->block_size;
    if (bs == 0) bs = QSF_BLOCK_SIZE;
    if (id == 0) id = hd * 4;

    /* Allocate working buffers */
    engine->activation = (float*)arena_alloc(engine->arena, hd * 4, 64, "activation");
    engine->residual   = (float*)arena_alloc(engine->arena, hd * 4, 64, "residual");
    engine->scratch    = (float*)arena_alloc(engine->arena, id * 4, 64, "scratch");
    engine->scratch2   = (float*)arena_alloc(engine->arena, id * 4, 64, "scratch2");
    engine->q_buf      = (float*)arena_alloc(engine->arena,
                          h->num_heads * engine->model.head_dim * 4, 64, "q_buf");
    engine->k_buf      = (float*)arena_alloc(engine->arena,
                          engine->model.num_kv_heads * engine->model.head_dim * 4, 64, "k_buf");
    engine->v_buf      = (float*)arena_alloc(engine->arena,
                          engine->model.num_kv_heads * engine->model.head_dim * 4, 64, "v_buf");
    engine->attn_out   = (float*)arena_alloc(engine->arena, hd * 4, 64, "attn_out");
    engine->attn_scores = (float*)arena_alloc(engine->arena,
                           h->num_heads * engine->budget.kv_max_seq * 4, 64, "attn_scores");
    engine->logits     = (float*)arena_alloc(engine->arena,
                          h->vocab_size * 4, 64, "logits");

    /* Layer buffers */
    engine->layer_buf = (uint8_t*)arena_alloc(engine->arena,
                         engine->budget.layer_buf_size, 64, "layer_buf");
    engine->layer_buf_compressed = (uint8_t*)arena_alloc(engine->arena,
                         engine->budget.layer_buf_size, 64, "layer_buf_comp");

    /* MoE buffers (only if model uses MoE) */
    if (h->num_experts > 1) {
        uint32_t ne  = h->num_experts;
        uint32_t nae = h->num_active_experts;
        if (nae == 0) nae = 2;  /* default top-2 */
        uint32_t expert_id = h->expert_intermediate_dim;
        if (expert_id == 0) expert_id = id;  /* fallback to global intermediate_dim */

        engine->router_logits  = (float*)arena_alloc(engine->arena,
                                   ne * sizeof(float), 64, "moe_router");
        engine->expert_out     = (float*)arena_alloc(engine->arena,
                                   hd * sizeof(float), 64, "moe_expert_out");
        engine->expert_scratch = (float*)arena_alloc(engine->arena,
                                   expert_id * sizeof(float), 64, "moe_expert_scratch");
        engine->expert_scratch2 = (float*)arena_alloc(engine->arena,
                                   expert_id * sizeof(float), 64, "moe_expert_scratch2");
        engine->expert_indices = (int*)arena_alloc(engine->arena,
                                   nae * sizeof(int), 64, "moe_expert_idx");
        engine->expert_weights = (float*)arena_alloc(engine->arena,
                                   nae * sizeof(float), 64, "moe_expert_wt");

        if (!engine->router_logits || !engine->expert_out ||
            !engine->expert_indices || !engine->expert_weights) {
            arena_destroy(engine->arena);
            qsf_model_free(&engine->model);
            return QSF_ERR_OUT_OF_MEMORY;
        }

        if (cfg->verbose >= 1) {
            fprintf(stderr, "[qstream] MoE: %u experts, top-%u active, "
                    "expert FFN dim=%u\n", ne, nae, expert_id);
        }
    }

    if (!engine->activation || !engine->residual || !engine->scratch ||
        !engine->q_buf || !engine->logits || !engine->layer_buf) {
        arena_destroy(engine->arena);
        qsf_model_free(&engine->model);
        return QSF_ERR_OUT_OF_MEMORY;
    }

    /* KV cache */
    err = alloc_kv_cache(&engine->kv_cache, engine->arena,
                          h->num_layers, engine->model.num_kv_heads,
                          engine->model.head_dim, engine->budget.kv_max_seq);
    if (err != QSF_OK) {
        arena_destroy(engine->arena);
        qsf_model_free(&engine->model);
        return err;
    }

    /* Load norm weights */
    err = load_norms(engine);
    if (err != QSF_OK) {
        arena_destroy(engine->arena);
        qsf_model_free(&engine->model);
        return err;
    }

    /* Load tokenizer */
    if (h->tokenizer_offset > 0) {
        size_t tok_max = engine->model.file.file_size - (size_t)h->tokenizer_offset;
        const void* tok_data = file_access_get(&engine->model.file,
                                                (size_t)h->tokenizer_offset, tok_max);
        if (tok_data) {
            qsf_tokenizer_load(&engine->tokenizer, tok_data, tok_max,
                                h->bos_token_id, h->eos_token_id,
                                h->pad_token_id, h->unk_token_id);
        }
    }

    /* Load position embeddings for GPT-2 / learned pos encoding */
    if (h->pos_enc_type == QSF_POS_LEARNED && h->embedding_offset > 0) {
        QSFEmbeddingHeader eh;
        const void* emb_hdr_data = file_access_get(&engine->model.file,
            (size_t)h->embedding_offset, sizeof(eh));
        if (emb_hdr_data) {
            memcpy(&eh, emb_hdr_data, sizeof(eh));

            /* Position embedding table follows vocab embeddings */
            size_t emb_quant_bytes = qsf_quant_block_size(
                (QSFQuantType)eh.quant_type, bs) *
                ((hd + bs - 1) / bs);
            size_t vocab_data_size = (size_t)eh.num_vectors * emb_quant_bytes;
            size_t pos_start = (size_t)h->embedding_offset + sizeof(QSFEmbeddingHeader);
            if (eh.num_chunks > 1) pos_start += eh.num_chunks * 8;
            pos_start += vocab_data_size;

            /* Allocate FP32 position embedding table */
            uint32_t max_pos = h->max_seq_len;
            if (max_pos == 0) max_pos = 2048;
            engine->pos_embed = (float*)arena_alloc(engine->arena,
                max_pos * hd * sizeof(float), 64, "pos_embed");

            if (engine->pos_embed) {
                /* Try to load and dequantize position embeddings */
                for (uint32_t pos = 0; pos < max_pos; pos++) {
                    size_t pos_offset = pos_start + pos * emb_quant_bytes;
                    const void* pos_data = file_access_get(&engine->model.file,
                        pos_offset, emb_quant_bytes);
                    if (pos_data) {
                        int num_blocks = (hd + bs - 1) / bs;
                        size_t blk_sz = qsf_quant_block_size(
                            (QSFQuantType)eh.quant_type, bs);
                        const uint8_t* bp = (const uint8_t*)pos_data;
                        float* dst = engine->pos_embed + pos * hd;
                        for (int b = 0; b < num_blocks; b++) {
                            int count = (b + 1 < num_blocks) ? (int)bs : (int)(hd - b * bs);
                            switch (eh.quant_type) {
                                case QSF_QUANT_4BIT_ASYM:
                                    qsf_dequant_block_4bit(bp, dst + b * bs, count); break;
                                case QSF_QUANT_4BIT_SYM:
                                    qsf_dequant_block_4bit_sym(bp, dst + b * bs, count); break;
                                default:
                                    qsf_dequant_block_4bit(bp, dst + b * bs, count); break;
                            }
                            bp += blk_sz;
                        }
                    } else {
                        memset(engine->pos_embed + pos * hd, 0, hd * sizeof(float));
                    }
                }
            }
        }
    }

    /* Load output head pointer (for tied or separate) */
    if (!engine->model.weight_tied && h->final_offset > 0) {
        /* Output head is after final norm in the final section */
        uint64_t off = h->final_offset + sizeof(QSFFinalHeader);
        uint32_t norm_size = hd * 2;  /* fp16 norm weights */
        if (engine->model.has_bias_norm) norm_size += hd * 2;
        off += norm_size;

        if (off < engine->model.file.file_size) {
            engine->output_head = (void*)file_access_get(
                &engine->model.file, (size_t)off,
                engine->model.file.file_size - (size_t)off);
            engine->output_head_quant = h->quant_type;
        }
    }

    /* Store engine config for runtime checks */
    engine->check_nan = cfg->check_nan;
    engine->verbose = cfg->verbose;

    if (cfg->verbose >= 1) {
        arena_stats(engine->arena, stderr);
    }

    engine->loaded = 1;
    return QSF_OK;
}

/* ── Single-token forward pass ───────────────────────────────────── */
QSFError qsf_forward(QSFEngine* engine, uint32_t token_id, int position) {
    const QSFHeader* h = &engine->model.header;
    const QSFKernelTable* k = &engine->kernels;
    uint32_t hd = h->hidden_dim;
    uint32_t nh = h->num_heads;
    uint32_t nkv = engine->model.num_kv_heads;
    uint32_t head_d = engine->model.head_dim;
    uint32_t id = h->intermediate_dim;
    if (id == 0) id = hd * 4;
    uint32_t bs = h->block_size;
    if (bs == 0) bs = QSF_BLOCK_SIZE;

    float* x = engine->activation;   /* current hidden state */
    float* res = engine->residual;

    /* ─── Embedding lookup ───────────────────────────────────────── */
    if (h->embedding_offset > 0 && token_id < h->vocab_size) {
        /* Get embedding for token_id */
        QSFEmbeddingHeader eh;
        const void* emb_hdr_data = file_access_get(&engine->model.file,
            (size_t)h->embedding_offset, sizeof(eh));
        if (emb_hdr_data) {
            memcpy(&eh, emb_hdr_data, sizeof(eh));

            /* Compute embedding size per token */
            size_t emb_quant_bytes = qsf_quant_block_size(
                (QSFQuantType)eh.quant_type, bs) *
                ((hd + bs - 1) / bs);

            size_t emb_data_offset = (size_t)h->embedding_offset +
                                     sizeof(QSFEmbeddingHeader);

            /* Skip chunk index if present */
            if (eh.num_chunks > 1) {
                emb_data_offset += eh.num_chunks * 8;
            }

            size_t token_offset = emb_data_offset + token_id * emb_quant_bytes;
            const void* emb_data = file_access_get(&engine->model.file,
                                                    token_offset, emb_quant_bytes);
            if (emb_data) {
                /* Dequantize embedding into activation buffer */
                int num_blocks = (hd + bs - 1) / bs;
                size_t blk_sz = qsf_quant_block_size(
                    (QSFQuantType)eh.quant_type, bs);
                const uint8_t* bp = (const uint8_t*)emb_data;
                for (int b = 0; b < num_blocks; b++) {
                    int count = (b + 1 < num_blocks) ? (int)bs : (int)(hd - b * bs);
                    switch (eh.quant_type) {
                        case QSF_QUANT_2BIT_ASYM:
                            qsf_dequant_block_2bit(bp, x + b * bs, count); break;
                        case QSF_QUANT_4BIT_ASYM:
                            qsf_dequant_block_4bit(bp, x + b * bs, count); break;
                        case QSF_QUANT_4BIT_SYM:
                            qsf_dequant_block_4bit_sym(bp, x + b * bs, count); break;
                        default:
                            qsf_dequant_block_4bit(bp, x + b * bs, count); break;
                    }
                    bp += blk_sz;
                }
            }
        }
    } else {
        memset(x, 0, hd * sizeof(float));
    }

    /* Add position embedding for GPT-2 */
    if (h->pos_enc_type == QSF_POS_LEARNED && engine->pos_embed) {
        for (uint32_t i = 0; i < hd; i++) {
            x[i] += engine->pos_embed[position * hd + i];
        }
    }

    /* Save residual */
    memcpy(res, x, hd * sizeof(float));

    /* ─── Transformer layers ─────────────────────────────────────── */
    for (uint32_t layer = 0; layer < h->num_layers; layer++) {
        if (engine->interrupted) return QSF_ERR_INTERNAL;

        QSFLayerIndex* li = &engine->model.layer_index[layer];

        /* Load layer data — decompress if needed */
        const uint8_t* layer_data;
        if (li->compression_type != QSF_COMPRESS_NONE) {
            /* LZ4/LZ4HC compressed: decompress into layer_buf */
            const void* decompressed = qsf_decompress_layer(
                &engine->model, layer,
                engine->layer_buf, engine->budget.layer_buf_size);
            if (!decompressed) {
                qsf_set_error(QSF_ERR_DECOMPRESSION, "layer decompression failed");
                return QSF_ERR_DECOMPRESSION;
            }
            layer_data = (const uint8_t*)decompressed;
        } else {
            layer_data = (const uint8_t*)file_access_get(
                &engine->model.file, (size_t)li->offset, li->compressed_size);
            if (!layer_data) return QSF_ERR_IO_FAILURE;
        }

        int nt = li->num_tensors;
        uint8_t layer_qt = li->quant_type;
        if (layer_qt == QSF_QUANT_USE_DEFAULT) layer_qt = h->quant_type;

        /* ── Attention norm ──────────────────────────────────────── */
        float* norm_w = engine->attn_norm_weights + layer * hd;
        float* norm_b = engine->attn_norm_biases ?
                         engine->attn_norm_biases + layer * hd : NULL;

        if (h->norm_type == QSF_NORM_RMSNORM_PRE || h->norm_type == QSF_NORM_RMSNORM_POST) {
            k->rms_norm(res, x, norm_w, norm_b, hd, h->norm_epsilon);
        } else {
            k->layer_norm(res, x, norm_w, norm_b, hd, h->norm_epsilon);
        }

        /* ── QKV Projection ──────────────────────────────────────── */
        QSFTensorHeader th;
        const void* tensor_data;

        if (h->attn_type == QSF_ATTN_FUSED_QKV) {
            /* Fused QKV: single matvec, then split */
            tensor_data = find_tensor(layer_data, nt, QSF_TENSOR_FUSED_QKV, &th);
            if (!tensor_data) {
                qsf_set_error(QSF_ERR_INVALID_MODEL, "missing fused QKV tensor");
                return QSF_ERR_INVALID_MODEL;
            }
            {
                uint8_t qt = (th.quant_type == QSF_QUANT_USE_DEFAULT) ? layer_qt : th.quant_type;
                float* qkv_buf = engine->scratch;
                get_matvec_fn(k, qt)(tensor_data, x, qkv_buf, th.rows, th.cols, bs);

                /* Split into Q, K, V */
                int q_dim = nh * head_d;
                int kv_dim = nkv * head_d;
                memcpy(engine->q_buf, qkv_buf, q_dim * sizeof(float));
                memcpy(engine->k_buf, qkv_buf + q_dim, kv_dim * sizeof(float));
                memcpy(engine->v_buf, qkv_buf + q_dim + kv_dim, kv_dim * sizeof(float));
            }
        } else {
            /* Separate Q, K, V projections */
            tensor_data = find_tensor(layer_data, nt, QSF_TENSOR_ATTN_Q, &th);
            if (!tensor_data) {
                qsf_set_error(QSF_ERR_INVALID_MODEL, "missing Q projection tensor");
                return QSF_ERR_INVALID_MODEL;
            }
            {
                uint8_t qt = (th.quant_type == QSF_QUANT_USE_DEFAULT) ? layer_qt : th.quant_type;
                get_matvec_fn(k, qt)(tensor_data, x, engine->q_buf, th.rows, th.cols, bs);
            }
            tensor_data = find_tensor(layer_data, nt, QSF_TENSOR_ATTN_K, &th);
            if (!tensor_data) {
                qsf_set_error(QSF_ERR_INVALID_MODEL, "missing K projection tensor");
                return QSF_ERR_INVALID_MODEL;
            }
            {
                uint8_t qt = (th.quant_type == QSF_QUANT_USE_DEFAULT) ? layer_qt : th.quant_type;
                get_matvec_fn(k, qt)(tensor_data, x, engine->k_buf, th.rows, th.cols, bs);
            }
            tensor_data = find_tensor(layer_data, nt, QSF_TENSOR_ATTN_V, &th);
            if (!tensor_data) {
                qsf_set_error(QSF_ERR_INVALID_MODEL, "missing V projection tensor");
                return QSF_ERR_INVALID_MODEL;
            }
            {
                uint8_t qt = (th.quant_type == QSF_QUANT_USE_DEFAULT) ? layer_qt : th.quant_type;
                get_matvec_fn(k, qt)(tensor_data, x, engine->v_buf, th.rows, th.cols, bs);
            }
        }

        /* ── RoPE ────────────────────────────────────────────────── */
        if (engine->model.uses_rope) {
            float theta = h->rope_theta;
            if (theta <= 0) theta = 10000.0f;
            apply_rope(engine->q_buf, engine->k_buf, head_d,
                       position, theta, nh, nkv);
        }

        /* ── Store K, V in cache ─────────────────────────────────── */
        kv_cache_store(&engine->kv_cache, layer, position,
                       engine->k_buf, engine->v_buf);

        /* ── Attention scores per head ───────────────────────────── */
        int seq_len = position + 1;
        int groups = nh / nkv;  /* GQA: how many Q heads share one K/V head */

        memset(engine->attn_out, 0, hd * sizeof(float));

        for (uint32_t head = 0; head < nh; head++) {
            int kv_head = head / groups;
            float* q = engine->q_buf + head * head_d;
            float* scores = engine->attn_scores + head * engine->budget.kv_max_seq;
            QSFKVHead* kv = &engine->kv_cache.heads[layer * nkv + kv_head];

            /* Compute Q · K^T / sqrt(d) for each position */
            for (int p = 0; p < seq_len; p++) {
                float* cached_k = (float*)kv->k_cache + p * head_d;
                scores[p] = k->dot(q, cached_k, head_d) / sqrtf((float)head_d);
            }

            /* Causal masking: positions > current are already absent */
            /* Softmax */
            k->softmax(scores, scores, seq_len);

            /* Weighted sum of V */
            float* out_head = engine->attn_out + head * head_d;
            for (int p = 0; p < seq_len; p++) {
                float* cached_v = (float*)kv->v_cache + p * head_d;
                float score = scores[p];
                for (uint32_t d = 0; d < head_d; d++) {
                    out_head[d] += score * cached_v[d];
                }
            }
        }

        /* ── Output projection ───────────────────────────────────── */
        tensor_data = find_tensor(layer_data, nt, QSF_TENSOR_ATTN_O, &th);
        if (!tensor_data) {
            qsf_set_error(QSF_ERR_INVALID_MODEL, "missing O projection tensor");
            return QSF_ERR_INVALID_MODEL;
        }
        {
            uint8_t qt = (th.quant_type == QSF_QUANT_USE_DEFAULT) ? layer_qt : th.quant_type;
            get_matvec_fn(k, qt)(tensor_data, engine->attn_out, x, th.rows, th.cols, bs);
        }

        /* ── Residual add ────────────────────────────────────────── */
        k->vec_add(res, x, res, hd);

        /* ── FFN norm ────────────────────────────────────────────── */
        float* fnorm_w = engine->ffn_norm_weights + layer * hd;
        float* fnorm_b = engine->ffn_norm_biases ?
                          engine->ffn_norm_biases + layer * hd : NULL;

        if (h->norm_type == QSF_NORM_RMSNORM_PRE || h->norm_type == QSF_NORM_RMSNORM_POST) {
            k->rms_norm(res, x, fnorm_w, fnorm_b, hd, h->norm_epsilon);
        } else {
            k->layer_norm(res, x, fnorm_w, fnorm_b, hd, h->norm_epsilon);
        }

        /* ── FFN ─────────────────────────────────────────────────── */
        if (h->ffn_type == QSF_FFN_MOE && h->num_experts > 1) {
            /* ── Mixture-of-Experts FFN ──────────────────────────── *
             * MoE replaces the dense FFN with sparse expert routing: *
             *   1. Router: normed_x → logits[num_experts]           *
             *   2. Softmax → top-K selection                        *
             *   3. Per-expert gated FFN, weighted accumulate        *
             * ─────────────────────────────────────────────────────── */
            uint32_t num_exp = h->num_experts;
            uint32_t num_active = h->num_active_experts;
            if (num_active == 0) num_active = 2;
            uint32_t eid = h->expert_intermediate_dim;
            if (eid == 0) eid = id;

            /* Save normed activation (x) — we'll zero x for accumulation
             * but need the normed value as input to every expert. */
            memcpy(engine->expert_out, x, hd * sizeof(float));

            /* 1. Router: normed_x → router_logits[num_experts] */
            tensor_data = find_tensor(layer_data, nt,
                                       QSF_TENSOR_MOE_ROUTER, &th);
            if (!tensor_data) {
                qsf_set_error(QSF_ERR_INVALID_MODEL, "missing MoE router");
                return QSF_ERR_INVALID_MODEL;
            }
            {
                uint8_t qt = (th.quant_type == QSF_QUANT_USE_DEFAULT)
                             ? layer_qt : th.quant_type;
                get_matvec_fn(k, qt)(tensor_data, engine->expert_out,
                    engine->router_logits, th.rows, th.cols, bs);
            }

            /* 2. Softmax over router logits */
            k->softmax(engine->router_logits, engine->router_logits,
                       (int)num_exp);

            /* 3. Top-K selection (insertion-sort into small array) */
            for (uint32_t i = 0; i < num_active; i++) {
                engine->expert_indices[i] = -1;
                engine->expert_weights[i] = -1e30f;
            }
            for (uint32_t e = 0; e < num_exp; e++) {
                float w = engine->router_logits[e];
                for (uint32_t i = 0; i < num_active; i++) {
                    if (w > engine->expert_weights[i]) {
                        for (uint32_t j = num_active - 1; j > i; j--) {
                            engine->expert_weights[j] =
                                engine->expert_weights[j - 1];
                            engine->expert_indices[j] =
                                engine->expert_indices[j - 1];
                        }
                        engine->expert_weights[i] = w;
                        engine->expert_indices[i] = (int)e;
                        break;
                    }
                }
            }

            /* 4. Renormalize top-K weights (Mixtral-style) */
            if (h->moe_norm_topk) {
                float wsum = 0.0f;
                for (uint32_t i = 0; i < num_active; i++)
                    wsum += engine->expert_weights[i];
                if (wsum > 0.0f) {
                    float inv = 1.0f / wsum;
                    for (uint32_t i = 0; i < num_active; i++)
                        engine->expert_weights[i] *= inv;
                }
            }

            /* 5. Zero output accumulator */
            memset(x, 0, hd * sizeof(float));

            /* 6. Execute each active expert (gated FFN) */
            for (uint32_t ai = 0; ai < num_active; ai++) {
                int eidx = engine->expert_indices[ai];
                if (eidx < 0) continue;
                float ew = engine->expert_weights[ai];

                /* Expert gate projection: normed_x → expert_scratch */
                tensor_data = find_tensor(layer_data, nt,
                    (QSFTensorType)QSF_MOE_GATE(eidx), &th);
                if (tensor_data) {
                    uint8_t qt = (th.quant_type == QSF_QUANT_USE_DEFAULT)
                                 ? layer_qt : th.quant_type;
                    get_matvec_fn(k, qt)(tensor_data, engine->expert_out,
                        engine->expert_scratch, th.rows, th.cols, bs);
                }

                /* Expert up projection: normed_x → expert_scratch2 */
                tensor_data = find_tensor(layer_data, nt,
                    (QSFTensorType)QSF_MOE_UP(eidx), &th);
                if (tensor_data) {
                    uint8_t qt = (th.quant_type == QSF_QUANT_USE_DEFAULT)
                                 ? layer_qt : th.quant_type;
                    get_matvec_fn(k, qt)(tensor_data, engine->expert_out,
                        engine->expert_scratch2, th.rows, th.cols, bs);
                }

                /* Activation on gate (SiLU for Mixtral/LLaMA-family) */
                if (h->activation == QSF_ACT_SILU) {
                    k->silu(engine->expert_scratch, (int)eid);
                } else if (h->activation == QSF_ACT_GELU ||
                           h->activation == QSF_ACT_GELU_APPROX) {
                    k->gelu(engine->expert_scratch, (int)eid);
                } else {
                    k->relu(engine->expert_scratch, (int)eid);
                }

                /* gate * up → expert_scratch */
                k->vec_mul(engine->expert_scratch, engine->expert_scratch2,
                           engine->expert_scratch, (int)eid);

                /* Expert down projection: expert_scratch → scratch
                 * (scratch is id-sized, guaranteed >= hd; not used
                 *  by dense FFN path since we're in MoE branch) */
                tensor_data = find_tensor(layer_data, nt,
                    (QSFTensorType)QSF_MOE_DOWN(eidx), &th);
                if (tensor_data) {
                    uint8_t qt = (th.quant_type == QSF_QUANT_USE_DEFAULT)
                                 ? layer_qt : th.quant_type;
                    get_matvec_fn(k, qt)(tensor_data, engine->expert_scratch,
                        engine->scratch, th.rows, th.cols, bs);
                }

                /* Weighted accumulate: x += weight * expert_output */
                for (uint32_t d = 0; d < hd; d++) {
                    x[d] += ew * engine->scratch[d];
                }
            }
        }

        /* ─── Dense FFN paths ────────────────────────────────────── */
        else if (h->ffn_type == QSF_FFN_GATED) {
            /* Gated FFN: gate_proj(x) * silu(up_proj(x)) → down_proj */

            /* Gate projection */
            tensor_data = find_tensor(layer_data, nt, QSF_TENSOR_FFN_GATE, &th);
            if (tensor_data) {
                uint8_t qt = (th.quant_type == QSF_QUANT_USE_DEFAULT) ? layer_qt : th.quant_type;
                get_matvec_fn(k, qt)(tensor_data, x, engine->scratch, th.rows, th.cols, bs);
            }

            /* Up projection */
            tensor_data = find_tensor(layer_data, nt, QSF_TENSOR_FFN_UP, &th);
            if (tensor_data) {
                uint8_t qt = (th.quant_type == QSF_QUANT_USE_DEFAULT) ? layer_qt : th.quant_type;
                get_matvec_fn(k, qt)(tensor_data, x, engine->scratch2, th.rows, th.cols, bs);
            }

            /* Activation on gate */
            int act_dim = id;
            if (h->activation == QSF_ACT_SILU) {
                k->silu(engine->scratch, act_dim);
            } else if (h->activation == QSF_ACT_GELU ||
                       h->activation == QSF_ACT_GELU_APPROX) {
                k->gelu(engine->scratch, act_dim);
            } else {
                k->relu(engine->scratch, act_dim);
            }

            /* gate * up */
            k->vec_mul(engine->scratch, engine->scratch2, engine->scratch, act_dim);

            /* Down projection */
            tensor_data = find_tensor(layer_data, nt, QSF_TENSOR_FFN_DOWN, &th);
            if (tensor_data) {
                uint8_t qt = (th.quant_type == QSF_QUANT_USE_DEFAULT) ? layer_qt : th.quant_type;
                get_matvec_fn(k, qt)(tensor_data, engine->scratch, x, th.rows, th.cols, bs);
            }
        } else {
            /* Standard FFN: up_proj → activation → down_proj */
            tensor_data = find_tensor(layer_data, nt, QSF_TENSOR_FFN_UP, &th);
            if (tensor_data) {
                uint8_t qt = (th.quant_type == QSF_QUANT_USE_DEFAULT) ? layer_qt : th.quant_type;
                get_matvec_fn(k, qt)(tensor_data, x, engine->scratch, th.rows, th.cols, bs);
            }

            int act_dim = id;
            if (h->activation == QSF_ACT_GELU ||
                h->activation == QSF_ACT_GELU_APPROX) {
                k->gelu(engine->scratch, act_dim);
            } else if (h->activation == QSF_ACT_SILU) {
                k->silu(engine->scratch, act_dim);
            } else {
                k->relu(engine->scratch, act_dim);
            }

            tensor_data = find_tensor(layer_data, nt, QSF_TENSOR_FFN_DOWN, &th);
            if (tensor_data) {
                uint8_t qt = (th.quant_type == QSF_QUANT_USE_DEFAULT) ? layer_qt : th.quant_type;
                get_matvec_fn(k, qt)(tensor_data, engine->scratch, x, th.rows, th.cols, bs);
            }
        }

        /* ── Residual add ────────────────────────────────────────── */
        k->vec_add(res, x, res, hd);

        /* ── NaN detection (if enabled) ─────────────────────────── */
        if (engine->check_nan) {
            for (uint32_t i = 0; i < hd; i++) {
                if (res[i] != res[i] || (res[i] > 1e30f || res[i] < -1e30f)) {
                    char buf[128];
                    snprintf(buf, sizeof(buf),
                             "NaN/Inf detected at layer %u, index %u", layer, i);
                    qsf_set_error(QSF_ERR_NAN_DETECTED, buf);
                    return QSF_ERR_NAN_DETECTED;
                }
            }
        }
    }

    /* ─── Final norm ─────────────────────────────────────────────── */
    if (h->norm_type == QSF_NORM_RMSNORM_PRE || h->norm_type == QSF_NORM_RMSNORM_POST) {
        k->rms_norm(res, x, engine->final_norm_weight, engine->final_norm_bias,
                    hd, h->norm_epsilon);
    } else {
        k->layer_norm(res, x, engine->final_norm_weight, engine->final_norm_bias,
                      hd, h->norm_epsilon);
    }

    /* ─── Output projection (logits) ─────────────────────────────── */
    if (engine->model.weight_tied) {
        /* Weight-tied: compute dot(x, embedding[v]) for each vocab entry */
        const QSFHeader* hh = h;
        if (hh->embedding_offset > 0) {
            QSFEmbeddingHeader eh;
            const void* emb_hdr = file_access_get(&engine->model.file,
                (size_t)hh->embedding_offset, sizeof(eh));
            if (emb_hdr) {
                memcpy(&eh, emb_hdr, sizeof(eh));
                size_t emb_quant_bytes = qsf_quant_block_size(
                    (QSFQuantType)eh.quant_type, bs) *
                    ((hd + bs - 1) / bs);
                size_t emb_data_offset = (size_t)hh->embedding_offset +
                    sizeof(QSFEmbeddingHeader);
                if (eh.num_chunks > 1) emb_data_offset += eh.num_chunks * 8;

                float* emb_row = engine->scratch;  /* reuse scratch as temp */
                for (uint32_t v = 0; v < hh->vocab_size; v++) {
                    size_t tok_off = emb_data_offset + v * emb_quant_bytes;
                    const void* emb_data = file_access_get(&engine->model.file,
                        tok_off, emb_quant_bytes);
                    if (emb_data) {
                        int num_blocks = (hd + bs - 1) / bs;
                        size_t blk_sz = qsf_quant_block_size(
                            (QSFQuantType)eh.quant_type, bs);
                        const uint8_t* bp = (const uint8_t*)emb_data;
                        for (int b = 0; b < num_blocks; b++) {
                            int cnt = (b + 1 < num_blocks) ? (int)bs : (int)(hd - b * bs);
                            switch (eh.quant_type) {
                                case QSF_QUANT_4BIT_ASYM:
                                    qsf_dequant_block_4bit(bp, emb_row + b*bs, cnt); break;
                                case QSF_QUANT_4BIT_SYM:
                                    qsf_dequant_block_4bit_sym(bp, emb_row + b*bs, cnt); break;
                                case QSF_QUANT_2BIT_ASYM:
                                    qsf_dequant_block_2bit(bp, emb_row + b*bs, cnt); break;
                                default:
                                    qsf_dequant_block_4bit(bp, emb_row + b*bs, cnt); break;
                            }
                            bp += blk_sz;
                        }
                        engine->logits[v] = k->dot(x, emb_row, hd);
                    } else {
                        engine->logits[v] = 0.0f;
                    }
                }
            } else {
                memset(engine->logits, 0, hh->vocab_size * sizeof(float));
            }
        } else {
            memset(engine->logits, 0, hh->vocab_size * sizeof(float));
        }
    } else if (engine->output_head) {
        /* Quantized output head matvec */
        uint8_t qt = (uint8_t)engine->output_head_quant;
        qsf_matvec_fn fn = get_matvec_fn(k, qt);
        fn(engine->output_head, x, engine->logits, h->vocab_size, hd, bs);
    } else {
        memset(engine->logits, 0, h->vocab_size * sizeof(float));
    }

    return QSF_OK;
}

/* ── Generate tokens ─────────────────────────────────────────────── */
QSFError qsf_engine_generate(QSFEngine* engine,
                              const char* prompt,
                              int max_new_tokens,
                              const QSFSamplingConfig* sampling,
                              qsf_token_callback callback,
                              void* userdata) {
    if (!engine || !engine->loaded) {
        return QSF_ERR_INVALID_INPUT;
    }

    const QSFHeader* h = &engine->model.header;
    QSFRng rng;
    qsf_rng_seed(&rng, sampling->seed);

    /* Tokenize prompt */
    uint32_t prompt_tokens[4096];
    int num_prompt = 0;

    /* Add BOS if the model has one */
    if (h->bos_token_id != QSF_NO_TOKEN) {
        prompt_tokens[num_prompt++] = h->bos_token_id;
    }

    if (prompt && strlen(prompt) > 0) {
        num_prompt += qsf_tokenizer_encode(&engine->tokenizer,
                                            prompt, strlen(prompt),
                                            prompt_tokens + num_prompt,
                                            4096 - num_prompt);
    }

    if (num_prompt == 0) {
        qsf_set_error(QSF_ERR_INVALID_INPUT, "empty prompt");
        return QSF_ERR_INVALID_INPUT;
    }

    /* Reset KV cache */
    engine->kv_cache.current_seq_len = 0;

    /* Process prompt tokens (prefill) */
    for (int i = 0; i < num_prompt; i++) {
        if (engine->interrupted) return QSF_OK;

        QSFError err = qsf_forward(engine, prompt_tokens[i], i);
        if (err != QSF_OK) return err;
        engine->kv_cache.current_seq_len = i + 1;
    }

    /* Generate new tokens */
    int position = num_prompt;

    /* Pre-allocate logits copy buffer (avoid malloc per token) */
    float* logits_copy = (float*)malloc(h->vocab_size * sizeof(float));
    if (!logits_copy) return QSF_ERR_OUT_OF_MEMORY;

    for (int t = 0; t < max_new_tokens; t++) {
        if (engine->interrupted) break;

        /* Sample next token from logits (copy to preserve original) */
        memcpy(logits_copy, engine->logits, h->vocab_size * sizeof(float));

        int next_token = qsf_sample(logits_copy, h->vocab_size, sampling, &rng);

        /* EOS check */
        if ((uint32_t)next_token == h->eos_token_id) break;

        /* Callback */
        if (callback) {
            const char* text = qsf_tokenizer_decode(&engine->tokenizer,
                                                     (uint32_t)next_token);
            if (callback((uint32_t)next_token, text, userdata)) break;
        }

        /* Check sequence length */
        if (position >= engine->budget.kv_max_seq - 1) {
            fprintf(stderr, "[qstream] max sequence length reached (%d)\n",
                    engine->budget.kv_max_seq);
            break;
        }

        /* Forward pass for generated token */
        QSFError err = qsf_forward(engine, (uint32_t)next_token, position);
        if (err != QSF_OK) { free(logits_copy); return err; }

        engine->kv_cache.current_seq_len = position + 1;
        position++;
    }

    free(logits_copy);
    return QSF_OK;
}

/* ── Interrupt ───────────────────────────────────────────────────── */
void qsf_engine_interrupt(QSFEngine* engine) {
    if (engine) engine->interrupted = 1;
}

/* ── Free ────────────────────────────────────────────────────────── */
void qsf_engine_free(QSFEngine* engine) {
    if (!engine) return;
    qsf_tokenizer_free(&engine->tokenizer);
    if (engine->arena) arena_destroy(engine->arena);
    qsf_model_free(&engine->model);
    memset(engine, 0, sizeof(*engine));
}
