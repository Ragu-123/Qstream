/*
 * QStream - platform.c
 * CPU feature detection, system RAM query, denormal flushing.
 */
#include "qsf/platform.h"
#include <string.h>
#include <stdio.h>

/* ── x86 CPUID ───────────────────────────────────────────────────── */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #ifdef _MSC_VER
    #include <intrin.h>
    static void cpuid(int info[4], int leaf) { __cpuid(info, leaf); }
    static void cpuid_count(int info[4], int leaf, int sub) { __cpuidex(info, leaf, sub); }
    static uint64_t xgetbv(uint32_t xcr) { return _xgetbv(xcr); }
  #else
    #include <cpuid.h>
    static void cpuid(int info[4], int leaf) {
        __cpuid(leaf, info[0], info[1], info[2], info[3]);
    }
    static void cpuid_count(int info[4], int leaf, int sub) {
        __cpuid_count(leaf, sub, info[0], info[1], info[2], info[3]);
    }
    #include <immintrin.h>
    static uint64_t xgetbv(uint32_t xcr) {
        uint32_t lo, hi;
        __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(xcr));
        return ((uint64_t)hi << 32) | lo;
    }
  #endif
  #define HAS_X86_CPUID 1
#endif

/* ── System RAM query ────────────────────────────────────────────── */
#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#elif defined(__linux__)
  #include <sys/sysinfo.h>
  #include <unistd.h>
#elif defined(__APPLE__)
  #include <sys/sysctl.h>
  #include <mach/mach.h>
#endif

uint64_t qsf_get_total_ram(void) {
#ifdef _WIN32
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    return ms.ullTotalPhys;
#elif defined(__linux__)
    struct sysinfo si;
    sysinfo(&si);
    return (uint64_t)si.totalram * si.mem_unit;
#elif defined(__APPLE__)
    int64_t ram = 0;
    size_t len = sizeof(ram);
    sysctlbyname("hw.memsize", &ram, &len, NULL, 0);
    return (uint64_t)ram;
#else
    return 0;
#endif
}

uint64_t qsf_get_available_ram(void) {
#ifdef _WIN32
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    return ms.ullAvailPhys;
#elif defined(__linux__)
    /* Read MemAvailable from /proc/meminfo — this includes reclaimable
     * page cache and buffers, giving a much more accurate picture than
     * sysinfo.freeram (which only reports truly free pages).
     * On a 12GB Colab, freeram might be 400MB while MemAvailable is 10GB. */
    {
        FILE* fp = fopen("/proc/meminfo", "r");
        if (fp) {
            char line[256];
            while (fgets(line, sizeof(line), fp)) {
                unsigned long long kb;
                if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
                    fclose(fp);
                    return (uint64_t)kb * 1024;
                }
            }
            fclose(fp);
        }
        /* Fallback to sysinfo if /proc/meminfo unavailable */
        struct sysinfo si;
        sysinfo(&si);
        /* Include buffers + cached as available */
        return (uint64_t)(si.freeram + si.bufferram) * si.mem_unit;
    }
#elif defined(__APPLE__)
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm;
    host_statistics64(mach_host_self(), HOST_VM_INFO64,
                      (host_info64_t)&vm, &count);
    return (uint64_t)(vm.free_count + vm.inactive_count) * 4096;
#else
    return 0;
#endif
}

/* ── Platform detection ──────────────────────────────────────────── */
void qsf_detect_platform(QSFPlatformInfo* info) {
    memset(info, 0, sizeof(*info));
    info->is_little_endian = qsf_is_little_endian();
    info->total_ram_bytes = qsf_get_total_ram();
    info->available_ram_bytes = qsf_get_available_ram();

#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    info->num_cores = (int)si.dwNumberOfProcessors;
#elif defined(__linux__) || defined(__APPLE__)
    info->num_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    if (info->num_cores < 1) info->num_cores = 1;

    info->best_isa = QSF_ISA_SCALAR;

#ifdef HAS_X86_CPUID
    int regs[4];

    /* Leaf 1: basic features */
    cpuid(regs, 1);
    info->has_sse42 = !!(regs[2] & (1 << 20));
    int os_xsave    = !!(regs[2] & (1 << 27));
    int has_avx_bit = !!(regs[2] & (1 << 28));
    info->has_fma   = !!(regs[2] & (1 << 12));
    info->has_f16c  = !!(regs[2] & (1 << 29));

    /* Check OS support for AVX state saving */
    int avx_os_support = 0;
    if (os_xsave && has_avx_bit) {
        uint64_t xcr0 = xgetbv(0);
        avx_os_support = (xcr0 & 0x6) == 0x6;  /* XMM + YMM state */
    }

    /* Leaf 7: extended features */
    cpuid_count(regs, 7, 0);
    info->has_avx2     = avx_os_support && !!(regs[1] & (1 << 5));
    info->has_avx512f  = avx_os_support && !!(regs[1] & (1 << 16));
    info->has_avx512bw = avx_os_support && !!(regs[1] & (1 << 30));

    if (info->has_avx512f) {
        /* Verify OS saves AVX-512 state (bit 5 + 6 + 7 of XCR0) */
        uint64_t xcr0 = xgetbv(0);
        if ((xcr0 & 0xE0) != 0xE0) {
            info->has_avx512f = 0;
            info->has_avx512bw = 0;
        }
    }

    info->has_avx = avx_os_support;

    /* Determine best ISA */
    if (info->has_avx512f && info->has_avx512bw) {
        info->best_isa = QSF_ISA_AVX512;
    } else if (info->has_avx2 && info->has_fma) {
        info->best_isa = QSF_ISA_AVX2;
    } else if (info->has_sse42) {
        info->best_isa = QSF_ISA_SSE42;
    }
#endif /* HAS_X86_CPUID */

#if defined(__ARM_NEON) || defined(__aarch64__)
    info->has_neon = 1;
    info->best_isa = QSF_ISA_NEON;
    #if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
        info->has_fp16_arith = 1;
    #endif
#endif
}

/* ── Denormal flushing (§4.5) ────────────────────────────────────── */
/*
 * Set FTZ (Flush To Zero, bit 15) + DAZ (Denormals Are Zero, bit 6)
 * in the MXCSR register. This prevents 10-100x slowdowns from denormal
 * floats in softmax, RMS norm, and accumulated rounding.
 * NOTE: per-thread! Must call in each worker thread.
 */
void qsf_flush_denormals(void) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    /* Both MSVC and GCC/Clang provide _mm_getcsr/_mm_setcsr via intrin.h / xmmintrin.h */
    unsigned int mxcsr = _mm_getcsr();
    mxcsr |= (1u << 15) | (1u << 6);  /* FTZ | DAZ */
    _mm_setcsr(mxcsr);
#elif defined(__aarch64__)
    uint64_t fpcr;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1 << 24);  /* FZ bit */
    __asm__ __volatile__("msr fpcr, %0" :: "r"(fpcr));
#endif
}
