/*
 * QStream - sampling.h
 * Token sampling strategies: temperature, top-k, top-p, greedy.
 */
#ifndef QSF_SAMPLING_H
#define QSF_SAMPLING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* xorshift128+ RNG state */
typedef struct {
    uint64_t s[2];
} QSFRng;

void   qsf_rng_seed(QSFRng* rng, uint64_t seed);
float  qsf_rng_float(QSFRng* rng);  /* uniform [0, 1) */

/* Sampling configuration */
typedef struct {
    float    temperature;   /* 0 = greedy, default 1.0 */
    int      top_k;         /* 0 = disabled, default 40 */
    float    top_p;         /* 1.0 = disabled, default 0.9 */
    float    repeat_penalty;/* 1.0 = disabled */
    uint64_t seed;          /* 0 = random */
} QSFSamplingConfig;

/* Initialize with sensible defaults */
void qsf_sampling_config_default(QSFSamplingConfig* cfg);

/* Apply temperature scaling in-place */
void qsf_apply_temperature(float* logits, int vocab_size, float temperature);

/* Top-k filtering: set logits below k-th largest to -INFINITY */
void qsf_top_k_filter(float* logits, int vocab_size, int k);

/* Top-p (nucleus) filtering */
void qsf_top_p_filter(float* logits, int vocab_size, float p);

/* Softmax (numerically stable) */
void qsf_softmax(const float* logits, float* probs, int n);

/* Sample from probability distribution */
int qsf_sample_categorical(const float* probs, int n, QSFRng* rng);

/* Argmax (greedy) */
int qsf_argmax(const float* logits, int n);

/* Full sampling pipeline: logits → token_id */
int qsf_sample(float* logits, int vocab_size,
               const QSFSamplingConfig* cfg, QSFRng* rng);

/* Apply repeat penalty to recent tokens */
void qsf_apply_repeat_penalty(float* logits, int vocab_size,
                               const uint32_t* recent_tokens, int num_recent,
                               float penalty);

#ifdef __cplusplus
}
#endif
#endif /* QSF_SAMPLING_H */
