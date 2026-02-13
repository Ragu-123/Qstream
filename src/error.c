/*
 * QStream - error.c
 * Thread-local error context implementation.
 */
#include "qsf/error.h"
#include <string.h>

#ifdef _MSC_VER
  #define THREAD_LOCAL __declspec(thread)
#else
  #define THREAD_LOCAL _Thread_local
#endif

static THREAD_LOCAL QSFError  tl_error = QSF_OK;
static THREAD_LOCAL char      tl_detail[256] = {0};

static const char* error_strings[] = {
    [QSF_OK]                  = "OK",
    [QSF_ERR_FILE_NOT_FOUND]  = "file not found or permission denied",
    [QSF_ERR_FILE_CORRUPTED]  = "file corrupted (CRC mismatch or bad data)",
    [QSF_ERR_FILE_VERSION]    = "unsupported file format version",
    [QSF_ERR_OUT_OF_MEMORY]   = "out of memory",
    [QSF_ERR_BUDGET_EXCEEDED] = "memory budget exceeded",
    [QSF_ERR_INVALID_MODEL]   = "invalid model (bad header or missing data)",
    [QSF_ERR_QUANT_MISMATCH]  = "quantization type mismatch",
    [QSF_ERR_IO_FAILURE]      = "I/O failure during read",
    [QSF_ERR_DECOMPRESSION]   = "decompression failure",
    [QSF_ERR_NAN_DETECTED]    = "NaN detected in activations",
    [QSF_ERR_INVALID_INPUT]   = "invalid input (null pointer or bad argument)",
    [QSF_ERR_SEQ_TOO_LONG]    = "sequence length exceeds model maximum",
    [QSF_ERR_UNSUPPORTED_ARCH]= "unsupported model architecture",
    [QSF_ERR_THREAD_FAILURE]  = "thread creation failed",
    [QSF_ERR_INTERNAL]        = "internal error (bug)",
};

const char* qsf_error_string(QSFError err) {
    if (err >= 0 && err < QSF_ERR_COUNT) {
        return error_strings[err];
    }
    return "unknown error";
}

void qsf_set_error(QSFError err, const char* detail) {
    tl_error = err;
    if (detail) {
        size_t len = strlen(detail);
        if (len >= sizeof(tl_detail)) len = sizeof(tl_detail) - 1;
        memcpy(tl_detail, detail, len);
        tl_detail[len] = '\0';
    } else {
        tl_detail[0] = '\0';
    }
}

QSFError qsf_get_error(void) {
    return tl_error;
}

const char* qsf_get_error_detail(void) {
    return tl_detail;
}

void qsf_clear_error(void) {
    tl_error = QSF_OK;
    tl_detail[0] = '\0';
}
