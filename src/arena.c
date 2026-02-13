#include "qstream.h"

#include <stdlib.h>
#include <string.h>

qs_arena_t *qs_arena_create(size_t size_bytes) {
  qs_arena_t *arena = (qs_arena_t *)calloc(1, sizeof(qs_arena_t));
  if (!arena) {
    return NULL;
  }
  arena->base = (uint8_t *)aligned_alloc(64, ((size_bytes + 63u) / 64u) * 64u);
  if (!arena->base) {
    free(arena);
    return NULL;
  }
  arena->total_size = ((size_bytes + 63u) / 64u) * 64u;
  arena->used = 0;
  return arena;
}

void qs_arena_destroy(qs_arena_t *arena) {
  if (!arena) {
    return;
  }
  free(arena->base);
  free(arena);
}

void *qs_arena_alloc(qs_arena_t *arena, size_t size, size_t alignment) {
  if (!arena || size == 0 || alignment == 0) {
    return NULL;
  }
  size_t mask = alignment - 1u;
  size_t aligned = (arena->used + mask) & ~mask;
  if (aligned + size > arena->total_size) {
    return NULL;
  }
  void *ptr = arena->base + aligned;
  arena->used = aligned + size;
  return ptr;
}

void qs_arena_reset(qs_arena_t *arena) {
  if (!arena) {
    return;
  }
  arena->used = 0;
  memset(arena->base, 0, arena->total_size);
}
