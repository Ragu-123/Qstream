#include "qstream.h"

#if defined(__x86_64__) || defined(__i386)
#include <cpuid.h>
#endif

qs_cpu_features_t qs_detect_cpu_features(void) {
  qs_cpu_features_t f = {0};
#if defined(__x86_64__) || defined(__i386)
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
  if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    f.has_sse42 = ((ecx >> 20u) & 1u) != 0u;
  }
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    f.has_avx2 = ((ebx >> 5u) & 1u) != 0u;
    f.has_avx512f = ((ebx >> 16u) & 1u) != 0u;
  }
#elif defined(__aarch64__) || defined(__arm__)
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  f.has_neon = 1;
#endif
#endif
  return f;
}
