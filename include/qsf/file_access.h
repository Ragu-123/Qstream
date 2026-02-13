/*
 * QStream - file_access.h
 * Abstraction over mmap and explicit I/O (pread / ReadFile).
 */
#ifndef QSF_FILE_ACCESS_H
#define QSF_FILE_ACCESS_H

#include "error.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void*    mmap_ptr;          /* NULL if using pread */
    size_t   file_size;
    int      use_mmap;

#ifdef _WIN32
    void*    file_handle;       /* HANDLE */
    void*    mapping_handle;    /* HANDLE */
#else
    int      fd;
#endif

    /* Fallback read buffer (for pread mode) */
    uint8_t* read_buf;
    size_t   read_buf_size;
} FileAccess;

QSFError    file_access_open(FileAccess* fa, const char* path, int allow_mmap);
const void* file_access_get(FileAccess* fa, size_t offset, size_t size);
void        file_access_prefetch(FileAccess* fa, size_t offset, size_t size);
void        file_access_release(FileAccess* fa, size_t offset, size_t size);
void        file_access_close(FileAccess* fa);

#ifdef __cplusplus
}
#endif
#endif /* QSF_FILE_ACCESS_H */
