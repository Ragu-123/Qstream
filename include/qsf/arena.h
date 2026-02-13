/*
 * QStream - arena.h
 * Zero-fragmentation bump-pointer arena allocator.
 */
#ifndef QSF_ARENA_H
#define QSF_ARENA_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef QSF_ARENA_MAX_LOG
#define QSF_ARENA_MAX_LOG 256
#endif

typedef struct {
    uint8_t* base;
    uint8_t* current;
    uint8_t* end;
    size_t   total_size;
    size_t   peak_used;
    int      num_allocations;

#ifndef NDEBUG
    struct {
        const char* name;
        void*       ptr;
        size_t      size;
        size_t      alignment;
    } alloc_log[QSF_ARENA_MAX_LOG];
    int alloc_count;
#endif
} Arena;

/* Create arena of given size. Returns NULL on failure. */
Arena*  arena_create(size_t size);

/* Allocate from arena with alignment. Returns NULL if full. */
void*   arena_alloc(Arena* arena, size_t size, size_t alignment, const char* name);

/* Reset arena (free all allocations at once). */
void    arena_reset(Arena* arena);

/* Destroy arena and release all memory. */
void    arena_destroy(Arena* arena);

/* Print allocation statistics to file stream. */
void    arena_stats(const Arena* arena, FILE* out);

/* How much space is used / remaining */
static inline size_t arena_used(const Arena* a) {
    return (size_t)(a->current - a->base);
}
static inline size_t arena_remaining(const Arena* a) {
    return (size_t)(a->end - a->current);
}

#ifdef __cplusplus
}
#endif
#endif /* QSF_ARENA_H */
