/*
 * QStream - error.h
 * Thread-safe error codes and context.
 */
#ifndef QSF_ERROR_H
#define QSF_ERROR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    QSF_OK = 0,
    QSF_ERR_FILE_NOT_FOUND,
    QSF_ERR_FILE_CORRUPTED,
    QSF_ERR_FILE_VERSION,
    QSF_ERR_OUT_OF_MEMORY,
    QSF_ERR_BUDGET_EXCEEDED,
    QSF_ERR_INVALID_MODEL,
    QSF_ERR_QUANT_MISMATCH,
    QSF_ERR_IO_FAILURE,
    QSF_ERR_DECOMPRESSION,
    QSF_ERR_NAN_DETECTED,
    QSF_ERR_INVALID_INPUT,
    QSF_ERR_SEQ_TOO_LONG,
    QSF_ERR_UNSUPPORTED_ARCH,
    QSF_ERR_THREAD_FAILURE,
    QSF_ERR_INTERNAL,
    QSF_ERR_COUNT
} QSFError;

/* Human-readable error string (static, do not free) */
const char* qsf_error_string(QSFError err);

/* Thread-local error context */
void        qsf_set_error(QSFError err, const char* detail);
QSFError    qsf_get_error(void);
const char* qsf_get_error_detail(void);
void        qsf_clear_error(void);

#ifdef __cplusplus
}
#endif
#endif /* QSF_ERROR_H */
