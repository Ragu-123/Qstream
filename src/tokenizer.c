/*
 * QStream - tokenizer.c
 * BPE tokenizer: encode text → token IDs, decode token IDs → text.
 */
#include "qsf/tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Load tokenizer from QSF binary data ─────────────────────────── */
QSFError qsf_tokenizer_load(QSFTokenizer* tok, const void* data, size_t size,
                             uint32_t bos, uint32_t eos, uint32_t pad, uint32_t unk) {
    memset(tok, 0, sizeof(*tok));

    if (size < sizeof(QSFTokenizerHeader)) {
        qsf_set_error(QSF_ERR_INVALID_MODEL, "tokenizer section too small");
        return QSF_ERR_INVALID_MODEL;
    }

    const uint8_t* p = (const uint8_t*)data;
    const uint8_t* data_end = p + size;
    QSFTokenizerHeader hdr;
    memcpy(&hdr, p, sizeof(hdr));
    p += sizeof(hdr);

    tok->type           = (QSFTokenizerType)hdr.tokenizer_type;
    tok->vocab_size     = hdr.vocab_size;
    tok->num_merges     = hdr.num_merges;
    tok->num_added_tokens = hdr.num_added_tokens;
    tok->bos_id         = bos;
    tok->eos_id         = eos;
    tok->pad_id         = pad;
    tok->unk_id         = unk;
    tok->byte_fallback  = !!(hdr.flags & 1);
    tok->add_prefix_space = !!(hdr.flags & 2);

    /* Allocate vocab */
    tok->vocab = (QSFVocabEntry*)calloc(tok->vocab_size, sizeof(QSFVocabEntry));
    if (!tok->vocab) {
        qsf_set_error(QSF_ERR_OUT_OF_MEMORY, "vocab alloc");
        return QSF_ERR_OUT_OF_MEMORY;
    }

    /* Read vocabulary entries with bounds checking */
    for (uint32_t i = 0; i < tok->vocab_size; i++) {
        if (p + 2 > data_end) {
            qsf_set_error(QSF_ERR_FILE_CORRUPTED, "tokenizer vocab truncated");
            qsf_tokenizer_free(tok);
            return QSF_ERR_FILE_CORRUPTED;
        }
        uint16_t len;
        memcpy(&len, p, 2); p += 2;
        if (p + len > data_end) {
            qsf_set_error(QSF_ERR_FILE_CORRUPTED, "tokenizer vocab entry truncated");
            qsf_tokenizer_free(tok);
            return QSF_ERR_FILE_CORRUPTED;
        }
        tok->vocab[i].length = len;
        tok->vocab[i].text = (char*)malloc(len + 1);
        if (!tok->vocab[i].text) {
            qsf_set_error(QSF_ERR_OUT_OF_MEMORY, "vocab entry alloc");
            qsf_tokenizer_free(tok);
            return QSF_ERR_OUT_OF_MEMORY;
        }
        memcpy(tok->vocab[i].text, p, len);
        tok->vocab[i].text[len] = '\0';
        p += len;
    }

    /* Skip vocab CRC32 */
    if (p + 4 > data_end) {
        qsf_set_error(QSF_ERR_FILE_CORRUPTED, "tokenizer CRC missing");
        qsf_tokenizer_free(tok);
        return QSF_ERR_FILE_CORRUPTED;
    }
    p += 4;

    /* Read merge rules */
    if (tok->num_merges > 0) {
        size_t merge_bytes = tok->num_merges * sizeof(QSFMergeRule);
        if (p + merge_bytes > data_end) {
            qsf_set_error(QSF_ERR_FILE_CORRUPTED, "tokenizer merges truncated");
            qsf_tokenizer_free(tok);
            return QSF_ERR_FILE_CORRUPTED;
        }
        tok->merges = (QSFMergeRule*)malloc(merge_bytes);
        if (!tok->merges) {
            qsf_set_error(QSF_ERR_OUT_OF_MEMORY, "merge alloc");
            qsf_tokenizer_free(tok);
            return QSF_ERR_OUT_OF_MEMORY;
        }
        memcpy(tok->merges, p, merge_bytes);
        p += merge_bytes;
    }

    /* Read added tokens */
    if (tok->num_added_tokens > 0) {
        tok->added_tokens = (QSFAddedToken*)calloc(tok->num_added_tokens,
                                                    sizeof(QSFAddedToken));
        if (!tok->added_tokens) {
            qsf_set_error(QSF_ERR_OUT_OF_MEMORY, "added tokens alloc");
            qsf_tokenizer_free(tok);
            return QSF_ERR_OUT_OF_MEMORY;
        }
        for (uint32_t i = 0; i < tok->num_added_tokens; i++) {
            if (p + 6 > data_end) {
                qsf_set_error(QSF_ERR_FILE_CORRUPTED, "added token truncated");
                qsf_tokenizer_free(tok);
                return QSF_ERR_FILE_CORRUPTED;
            }
            uint32_t id; memcpy(&id, p, 4); p += 4;
            uint16_t len; memcpy(&len, p, 2); p += 2;
            if (p + len + 1 > data_end) {
                qsf_set_error(QSF_ERR_FILE_CORRUPTED, "added token text truncated");
                qsf_tokenizer_free(tok);
                return QSF_ERR_FILE_CORRUPTED;
            }
            tok->added_tokens[i].id = id;
            tok->added_tokens[i].length = len;
            tok->added_tokens[i].text = (char*)malloc(len + 1);
            if (tok->added_tokens[i].text) {
                memcpy(tok->added_tokens[i].text, p, len);
                tok->added_tokens[i].text[len] = '\0';
            }
            p += len;
            tok->added_tokens[i].flags = *p; p += 1;
        }
    }

    return QSF_OK;
}

/* ── Simple BPE encode ───────────────────────────────────────────── */
/*
 * Minimal BPE: split text into UTF-8 bytes, then iteratively merge
 * the highest-priority pair. This is a simple O(n*m) implementation.
 */

/* Linked list node for BPE encoding */
typedef struct BPENode {
    uint32_t token_id;
    struct BPENode* next;
} BPENode;

int qsf_tokenizer_encode(const QSFTokenizer* tok,
                          const char* text, size_t text_len,
                          uint32_t* out_tokens, int max_tokens) {
    if (!tok || !text || text_len == 0 || !out_tokens || max_tokens <= 0)
        return 0;

    /* Step 1: Initialize each byte as its own token.
       For BPE with byte_fallback, each byte maps to a byte-level token.
       For GPT-2 BPE, each byte maps to its byte-level token (IDs 0-255). */
    int num_chars = (int)text_len;

    /* Build initial token list: look up single-character tokens */
    BPENode* nodes = (BPENode*)calloc(num_chars, sizeof(BPENode));
    if (!nodes) return 0;

    for (int i = 0; i < num_chars; i++) {
        /* Find single-char token in vocab */
        uint32_t best = tok->unk_id;
        unsigned char ch = (unsigned char)text[i];

        /* Search for single-byte token in full vocab */
        for (uint32_t v = 0; v < tok->vocab_size; v++) {
            if (tok->vocab[v].length == 1 &&
                (unsigned char)tok->vocab[v].text[0] == ch) {
                best = v;
                break;
            }
        }
        nodes[i].token_id = best;
        nodes[i].next = (i + 1 < num_chars) ? &nodes[i + 1] : NULL;
    }

    /* Step 2: Iterate merges in priority order */
    for (uint32_t m = 0; m < tok->num_merges; m++) {
        uint32_t a = tok->merges[m].token_a;
        uint32_t b = tok->merges[m].token_b;
        uint32_t merged = tok->merges[m].merged;

        /* Scan linked list for adjacent (a, b) pairs */
        BPENode* cur = &nodes[0];
        if (cur->token_id == (uint32_t)-1) {
            /* Find first valid node */
            while (cur && cur->token_id == (uint32_t)-1) cur = cur->next;
        }
        while (cur && cur->next) {
            BPENode* nxt = cur->next;
            /* Skip deleted nodes */
            while (nxt && nxt->token_id == (uint32_t)-1) nxt = nxt->next;
            if (!nxt) break;

            if (cur->token_id == a && nxt->token_id == b) {
                cur->token_id = merged;
                cur->next = nxt->next;
                nxt->token_id = (uint32_t)-1;  /* mark deleted */
            } else {
                cur = nxt;
            }
        }
    }

    /* Step 3: Collect output tokens */
    int count = 0;
    BPENode* cur = &nodes[0];
    while (cur && count < max_tokens) {
        if (cur->token_id != (uint32_t)-1) {
            out_tokens[count++] = cur->token_id;
        }
        cur = cur->next;
    }

    free(nodes);
    return count;
}

/* ── Decode ──────────────────────────────────────────────────────── */
const char* qsf_tokenizer_decode(const QSFTokenizer* tok, uint32_t token_id) {
    if (!tok || token_id >= tok->vocab_size) return "";
    return tok->vocab[token_id].text ? tok->vocab[token_id].text : "";
}

/* ── Free ────────────────────────────────────────────────────────── */
void qsf_tokenizer_free(QSFTokenizer* tok) {
    if (!tok) return;
    if (tok->vocab) {
        for (uint32_t i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i].text);
        }
        free(tok->vocab);
    }
    free(tok->merges);
    if (tok->added_tokens) {
        for (uint32_t i = 0; i < tok->num_added_tokens; i++) {
            free(tok->added_tokens[i].text);
        }
        free(tok->added_tokens);
    }
    memset(tok, 0, sizeof(*tok));
}
