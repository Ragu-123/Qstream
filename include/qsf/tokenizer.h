/*
 * QStream - tokenizer.h
 * BPE tokenizer: encode text → token IDs, decode token IDs → text.
 */
#ifndef QSF_TOKENIZER_H
#define QSF_TOKENIZER_H

#include "types.h"
#include "error.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* In-memory token entry */
typedef struct {
    char*    text;
    uint16_t length;
} QSFVocabEntry;

/* In-memory added token */
typedef struct {
    uint32_t id;
    char*    text;
    uint16_t length;
    uint8_t  flags;
} QSFAddedToken;

typedef struct {
    QSFTokenizerType type;
    uint32_t         vocab_size;

    /* Vocabulary: array indexed by token ID */
    QSFVocabEntry*   vocab;

    /* BPE merge rules (priority order, highest first) */
    QSFMergeRule*    merges;
    uint32_t         num_merges;

    /* Added / special tokens */
    QSFAddedToken*   added_tokens;
    uint32_t         num_added_tokens;

    /* Special token IDs */
    uint32_t bos_id;
    uint32_t eos_id;
    uint32_t pad_id;
    uint32_t unk_id;

    /* Flags */
    int byte_fallback;
    int add_prefix_space;
} QSFTokenizer;

/* Load tokenizer from QSF file at given offset */
QSFError qsf_tokenizer_load(QSFTokenizer* tok, const void* data, size_t size,
                             uint32_t bos, uint32_t eos, uint32_t pad, uint32_t unk);

/* Encode UTF-8 text into token IDs. Returns number of tokens. */
int      qsf_tokenizer_encode(const QSFTokenizer* tok,
                               const char* text, size_t text_len,
                               uint32_t* out_tokens, int max_tokens);

/* Decode a single token ID to string. Returns pointer to static string. */
const char* qsf_tokenizer_decode(const QSFTokenizer* tok, uint32_t token_id);

/* Free tokenizer resources */
void     qsf_tokenizer_free(QSFTokenizer* tok);

#ifdef __cplusplus
}
#endif
#endif /* QSF_TOKENIZER_H */
