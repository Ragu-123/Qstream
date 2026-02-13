/*
 * QStream - arena.c
 * Bump-pointer arena allocator.
 */
#include "qsf/arena.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <sys/mman.h>
#endif

#define ARENA_LARGE_THRESHOLD (1024 * 1024)  /* 1 MB */

Arena* arena_create(size_t size) {
    Arena* arena = (Arena*)calloc(1, sizeof(Arena));
    if (!arena) return NULL;

    /* Round up to 64-byte alignment */
    size = (size + 63) & ~(size_t)63;

    uint8_t* base = NULL;

#ifdef _WIN32
    if (size >= ARENA_LARGE_THRESHOLD) {
        base = (uint8_t*)VirtualAlloc(NULL, size,
                                       MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    }
    if (!base) {
        base = (uint8_t*)_aligned_malloc(size, 64);
    }
#else
    if (size >= ARENA_LARGE_THRESHOLD) {
        base = (uint8_t*)mmap(NULL, size, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) base = NULL;
    }
    if (!base) {
        if (posix_memalign((void**)&base, 64, size) != 0) base = NULL;
    }
#endif

    if (!base) {
        free(arena);
        return NULL;
    }

    memset(base, 0, size);
    arena->base = base;
    arena->current = base;
    arena->end = base + size;
    arena->total_size = size;
    arena->peak_used = 0;
    arena->num_allocations = 0;

#ifndef NDEBUG
    arena->alloc_count = 0;
#endif

    return arena;
}

void* arena_alloc(Arena* arena, size_t size, size_t alignment, const char* name) {
    if (!arena || size == 0) return NULL;

    /* Align current pointer */
    uintptr_t cur = (uintptr_t)arena->current;
    uintptr_t aligned = (cur + alignment - 1) & ~(alignment - 1);
    uint8_t* result = (uint8_t*)aligned;

    if (result + size > arena->end) {
        return NULL;  /* Budget exceeded */
    }

    arena->current = result + size;
    arena->num_allocations++;

    size_t used = (size_t)(arena->current - arena->base);
    if (used > arena->peak_used) {
        arena->peak_used = used;
    }

#ifndef NDEBUG
    if (arena->alloc_count < QSF_ARENA_MAX_LOG) {
        arena->alloc_log[arena->alloc_count].name = name;
        arena->alloc_log[arena->alloc_count].ptr = result;
        arena->alloc_log[arena->alloc_count].size = size;
        arena->alloc_log[arena->alloc_count].alignment = alignment;
        arena->alloc_count++;
    }
#endif
    (void)name;  /* suppress unused warning in release */

    return result;
}

void arena_reset(Arena* arena) {
    if (!arena) return;
    arena->current = arena->base;
    arena->num_allocations = 0;
#ifndef NDEBUG
    arena->alloc_count = 0;
#endif
}

void arena_destroy(Arena* arena) {
    if (!arena) return;

#ifdef _WIN32
    if (arena->total_size >= ARENA_LARGE_THRESHOLD) {
        VirtualFree(arena->base, 0, MEM_RELEASE);
    } else {
        _aligned_free(arena->base);
    }
#else
    if (arena->total_size >= ARENA_LARGE_THRESHOLD) {
        munmap(arena->base, arena->total_size);
    } else {
        free(arena->base);
    }
#endif

    arena->base = NULL;
    arena->current = NULL;
    arena->end = NULL;
    free(arena);
}

void arena_stats(const Arena* arena, FILE* out) {
    if (!arena || !out) return;

    size_t used = arena_used(arena);
    double pct = arena->total_size > 0
                 ? 100.0 * (double)used / (double)arena->total_size
                 : 0.0;

    fprintf(out, "=== Arena Statistics ===\n");
    fprintf(out, "  Total:       %8zu KB\n", arena->total_size / 1024);
    fprintf(out, "  Used:        %8zu KB (%.1f%%)\n", used / 1024, pct);
    fprintf(out, "  Peak:        %8zu KB\n", arena->peak_used / 1024);
    fprintf(out, "  Allocations: %d\n", arena->num_allocations);

#ifndef NDEBUG
    if (arena->alloc_count > 0) {
        fprintf(out, "  --- Allocation Log ---\n");
        for (int i = 0; i < arena->alloc_count; i++) {
            fprintf(out, "    %-24s %8zu bytes  (align %zu)\n",
                    arena->alloc_log[i].name ? arena->alloc_log[i].name : "?",
                    arena->alloc_log[i].size,
                    arena->alloc_log[i].alignment);
        }
    }
#endif
}
