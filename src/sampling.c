/*
 * QStream - sampling.c
 * Token sampling: temperature, top-k, top-p, categorical, argmax.
 */
#include "qsf/sampling.h"
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/* ── xorshift128+ RNG ────────────────────────────────────────────── */
void qsf_rng_seed(QSFRng* rng, uint64_t seed) {
    /* SplitMix64 to seed from a single value */
    seed += 0x9e3779b97f4a7c15ULL;
    seed = (seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9ULL;
    seed = (seed ^ (seed >> 27)) * 0x94d049bb133111ebULL;
    rng->s[0] = seed ^ (seed >> 31);
    if (rng->s[0] == 0) rng->s[0] = 1;
    seed += 0x9e3779b97f4a7c15ULL;
    seed = (seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9ULL;
    seed = (seed ^ (seed >> 27)) * 0x94d049bb133111ebULL;
    rng->s[1] = seed ^ (seed >> 31);
    if (rng->s[1] == 0) rng->s[1] = 1;
}

float qsf_rng_float(QSFRng* rng) {
    uint64_t s0 = rng->s[0];
    uint64_t s1 = rng->s[1];
    uint64_t result = s0 + s1;

    s1 ^= s0;
    rng->s[0] = ((s0 << 55) | (s0 >> 9)) ^ s1 ^ (s1 << 14);
    rng->s[1] = (s1 << 36) | (s1 >> 28);

    /* Convert upper 23 bits to [0, 1) float */
    return (float)(result >> 41) * (1.0f / (float)(1ULL << 23));
}

/* ── Config defaults ─────────────────────────────────────────────── */
void qsf_sampling_config_default(QSFSamplingConfig* cfg) {
    cfg->temperature    = 0.7f;
    cfg->top_k          = 40;
    cfg->top_p          = 0.9f;
    cfg->repeat_penalty = 1.0f;
    cfg->seed           = 42;
}

/* ── Temperature scaling ─────────────────────────────────────────── */
void qsf_apply_temperature(float* logits, int vocab_size, float temperature) {
    if (temperature <= 0.01f || temperature == 1.0f) return;
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= inv_temp;
    }
}

/* ── Repeat penalty ──────────────────────────────────────────────── */
void qsf_apply_repeat_penalty(float* logits, int vocab_size,
                               const uint32_t* recent_tokens, int num_recent,
                               float penalty) {
    if (penalty <= 1.0f || num_recent <= 0) return;
    for (int i = 0; i < num_recent; i++) {
        uint32_t tid = recent_tokens[i];
        if (tid < (uint32_t)vocab_size) {
            /* Divide positive logits, multiply negative logits by penalty */
            if (logits[tid] > 0) {
                logits[tid] /= penalty;
            } else {
                logits[tid] *= penalty;
            }
        }
    }
}

/* ── Top-K filtering ─────────────────────────────────────────────── */
void qsf_top_k_filter(float* logits, int vocab_size, int k) {
    if (k <= 0 || k >= vocab_size) return;

    /* Find k-th largest using partial selection */
    float* threshold_buf = (float*)malloc(vocab_size * sizeof(float));
    if (!threshold_buf) return;
    memcpy(threshold_buf, logits, vocab_size * sizeof(float));

    /* QuickSelect to find k-th element */
    int lo = 0, hi = vocab_size - 1;
    int target = k - 1;  /* 0-indexed k-th largest */

    while (lo < hi) {
        float pivot = threshold_buf[hi];
        int store = lo;
        for (int i = lo; i < hi; i++) {
            if (threshold_buf[i] > pivot) {  /* descending */
                float tmp = threshold_buf[store];
                threshold_buf[store] = threshold_buf[i];
                threshold_buf[i] = tmp;
                store++;
            }
        }
        float tmp = threshold_buf[store];
        threshold_buf[store] = threshold_buf[hi];
        threshold_buf[hi] = tmp;

        if (store == target) break;
        else if (store < target) lo = store + 1;
        else hi = store - 1;
    }

    float thresh = threshold_buf[target];
    free(threshold_buf);

    /* Filter: set everything below threshold to -inf */
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] < thresh) {
            logits[i] = -INFINITY;
        }
    }
}

/* ── Top-P (nucleus) filtering ───────────────────────────────────── */
void qsf_top_p_filter(float* logits, int vocab_size, float p) {
    if (p >= 1.0f) return;

    /* Compute softmax first */
    float* probs = (float*)malloc(vocab_size * sizeof(float));
    int*   idx   = (int*)malloc(vocab_size * sizeof(int));
    if (!probs || !idx) { free(probs); free(idx); return; }

    /* Softmax */
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++)
        if (logits[i] > max_val) max_val = logits[i];

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
        idx[i] = i;
    }
    if (sum > 0) {
        float inv = 1.0f / sum;
        for (int i = 0; i < vocab_size; i++) probs[i] *= inv;
    }

    /* Sort indices by probability descending (insertion sort — small N after top-k) */
    for (int i = 1; i < vocab_size; i++) {
        int key_idx = idx[i];
        float key_prob = probs[key_idx];
        int j = i - 1;
        while (j >= 0 && probs[idx[j]] < key_prob) {
            idx[j + 1] = idx[j];
            j--;
        }
        idx[j + 1] = key_idx;
    }

    /* Cumulative sum, mark tokens beyond p */
    float cum = 0.0f;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        cum += probs[idx[i]];
        if (cum > p) {
            cutoff = i + 1;  /* keep one past threshold */
            break;
        }
    }

    /* Zero out tokens beyond cutoff */
    for (int i = cutoff; i < vocab_size; i++) {
        logits[idx[i]] = -INFINITY;
    }

    free(probs);
    free(idx);
}

/* ── Softmax ─────────────────────────────────────────────────────── */
void qsf_softmax(const float* logits, float* probs, int n) {
    if (n <= 0) return;
    float max_val = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > max_val) max_val = logits[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }
    if (sum < 1e-30f) {
        float u = 1.0f / (float)n;
        for (int i = 0; i < n; i++) probs[i] = u;
        return;
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) probs[i] *= inv;
}

/* ── Categorical sampling ────────────────────────────────────────── */
int qsf_sample_categorical(const float* probs, int n, QSFRng* rng) {
    float r = qsf_rng_float(rng);
    float cum = 0.0f;
    for (int i = 0; i < n; i++) {
        cum += probs[i];
        if (cum >= r) return i;
    }
    /* Rounding edge case: return last non-zero token */
    for (int i = n - 1; i >= 0; i--) {
        if (probs[i] > 0.0f) return i;
    }
    return 0;
}

/* ── Argmax ──────────────────────────────────────────────────────── */
int qsf_argmax(const float* logits, int n) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > best_val) {
            best = i;
            best_val = logits[i];
        }
    }
    return best;
}

/* ── Full sampling pipeline ──────────────────────────────────────── */
int qsf_sample(float* logits, int vocab_size,
               const QSFSamplingConfig* cfg, QSFRng* rng) {
    /* Greedy (temperature=0) */
    if (cfg->temperature < 0.01f) {
        return qsf_argmax(logits, vocab_size);
    }

    /* Temperature */
    qsf_apply_temperature(logits, vocab_size, cfg->temperature);

    /* Top-K */
    if (cfg->top_k > 0 && cfg->top_k < vocab_size) {
        qsf_top_k_filter(logits, vocab_size, cfg->top_k);
    }

    /* Top-P */
    if (cfg->top_p > 0.0f && cfg->top_p < 1.0f) {
        qsf_top_p_filter(logits, vocab_size, cfg->top_p);
    }

    /* Softmax in-place on the logits buffer, then sample directly */
    qsf_softmax(logits, logits, vocab_size);
    return qsf_sample_categorical(logits, vocab_size, rng);
}
