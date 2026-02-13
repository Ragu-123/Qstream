/*
 * QStream - file_access.c
 * mmap / pread file access abstraction (Windows & POSIX).
 */
#include "qsf/file_access.h"
#include "qsf/error.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

QSFError file_access_open(FileAccess* fa, const char* path, int allow_mmap) {
    memset(fa, 0, sizeof(*fa));

#ifdef _WIN32
    /* Open file */
    HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) {
        qsf_set_error(QSF_ERR_FILE_NOT_FOUND, path);
        return QSF_ERR_FILE_NOT_FOUND;
    }

    /* Get file size */
    LARGE_INTEGER li;
    if (!GetFileSizeEx(fh, &li)) {
        CloseHandle(fh);
        qsf_set_error(QSF_ERR_IO_FAILURE, "GetFileSizeEx failed");
        return QSF_ERR_IO_FAILURE;
    }
    fa->file_size = (size_t)li.QuadPart;
    fa->file_handle = fh;

    /* Try mmap */
    if (allow_mmap && fa->file_size > 0) {
        HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
        if (mh) {
            void* ptr = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
            if (ptr) {
                fa->mmap_ptr = ptr;
                fa->mapping_handle = mh;
                fa->use_mmap = 1;
                return QSF_OK;
            }
            CloseHandle(mh);
        }
    }

    /* Fallback: allocate read buffer */
    fa->use_mmap = 0;
    fa->mapping_handle = NULL;
    fa->read_buf = NULL;
    fa->read_buf_size = 0;

#else /* POSIX */
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        qsf_set_error(QSF_ERR_FILE_NOT_FOUND, path);
        return QSF_ERR_FILE_NOT_FOUND;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        qsf_set_error(QSF_ERR_IO_FAILURE, "fstat failed");
        return QSF_ERR_IO_FAILURE;
    }
    fa->file_size = (size_t)st.st_size;
    fa->fd = fd;

    /* Try mmap */
    if (allow_mmap && fa->file_size > 0) {
        void* ptr = mmap(NULL, fa->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr != MAP_FAILED) {
            fa->mmap_ptr = ptr;
            fa->use_mmap = 1;
            madvise(ptr, fa->file_size, MADV_RANDOM);
            return QSF_OK;
        }
    }

    /* Fallback */
    fa->use_mmap = 0;
    fa->read_buf = NULL;
    fa->read_buf_size = 0;
#endif

    return QSF_OK;
}

const void* file_access_get(FileAccess* fa, size_t offset, size_t size) {
    if (offset + size > fa->file_size) return NULL;

    if (fa->use_mmap) {
        return (const uint8_t*)fa->mmap_ptr + offset;
    }

    /* pread fallback: ensure read buffer is large enough */
    if (fa->read_buf_size < size) {
        free(fa->read_buf);
        fa->read_buf = (uint8_t*)malloc(size);
        if (!fa->read_buf) { fa->read_buf_size = 0; return NULL; }
        fa->read_buf_size = size;
    }

#ifdef _WIN32
    LARGE_INTEGER li;
    li.QuadPart = (LONGLONG)offset;
    SetFilePointerEx(fa->file_handle, li, NULL, FILE_BEGIN);
    DWORD bytes_read = 0;
    if (!ReadFile(fa->file_handle, fa->read_buf, (DWORD)size, &bytes_read, NULL)
        || bytes_read != (DWORD)size) {
        return NULL;
    }
#else
    ssize_t ret = pread(fa->fd, fa->read_buf, size, (off_t)offset);
    if (ret < 0 || (size_t)ret != size) return NULL;
#endif

    return fa->read_buf;
}

void file_access_prefetch(FileAccess* fa, size_t offset, size_t size) {
    if (!fa->use_mmap) return;
    if (offset + size > fa->file_size) return;

#ifdef _WIN32
    /* PrefetchVirtualMemory available on Windows 8+ */
    /* Skipping for simplicity — mmap on Windows uses OS cache */
    (void)fa; (void)offset; (void)size;
#else
    madvise((uint8_t*)fa->mmap_ptr + offset, size, MADV_WILLNEED);
#endif
}

void file_access_release(FileAccess* fa, size_t offset, size_t size) {
    if (!fa->use_mmap) return;
#ifndef _WIN32
    madvise((uint8_t*)fa->mmap_ptr + offset, size, MADV_DONTNEED);
#else
    (void)fa; (void)offset; (void)size;
#endif
}

void file_access_close(FileAccess* fa) {
    if (!fa) return;

#ifdef _WIN32
    if (fa->mmap_ptr)        UnmapViewOfFile(fa->mmap_ptr);
    if (fa->mapping_handle)  CloseHandle(fa->mapping_handle);
    if (fa->file_handle)     CloseHandle(fa->file_handle);
#else
    if (fa->mmap_ptr && fa->use_mmap) munmap(fa->mmap_ptr, fa->file_size);
    if (fa->fd >= 0) close(fa->fd);
#endif

    free(fa->read_buf);
    memset(fa, 0, sizeof(*fa));

#ifndef _WIN32
    fa->fd = -1;
#endif
}
