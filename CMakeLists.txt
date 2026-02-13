cmake_minimum_required(VERSION 3.16)
project(qstream C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type defaults
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── Compiler flags ───────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4 /O2 /fp:fast)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG)
    else()
        add_compile_options(-O0 -g)
    endif()
    add_compile_options(-ffast-math)
endif()

# ── Include directories ─────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)

# ── Source files ─────────────────────────────────────────────────────
set(QSF_SOURCES
    src/error.c
    src/platform.c
    src/arena.c
    src/file_access.c
    src/format.c
    src/quant.c
    src/tokenizer.c
    src/kernels.c
    src/kernels_scalar.c
    src/sampling.c
    src/engine.c
)

# ── Platform-specific SIMD sources ───────────────────────────────────
# AVX2 kernels (x86/x64 only)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i[3-6]86)")
    set(AVX2_SOURCE src/kernels_avx2.c)

    if(MSVC)
        set_source_files_properties(${AVX2_SOURCE}
            PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    else()
        set_source_files_properties(${AVX2_SOURCE}
            PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mf16c")
    endif()

    list(APPEND QSF_SOURCES ${AVX2_SOURCE})
endif()

# NEON kernels (ARM64)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|ARM64)")
    list(APPEND QSF_SOURCES src/kernels_neon.c)
    # NEON is always available on ARM64, no special flags needed
endif()

# ── Static library ──────────────────────────────────────────────────
add_library(qsf_lib STATIC ${QSF_SOURCES})
target_include_directories(qsf_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# ── CLI executable ──────────────────────────────────────────────────
add_executable(qstream src/main.c)
target_link_libraries(qstream PRIVATE qsf_lib)

# ── Test executable ─────────────────────────────────────────────────
add_executable(qstream_tests tests/test_core.c)
target_link_libraries(qstream_tests PRIVATE qsf_lib)

# Link math library on Unix
if(UNIX)
    target_link_libraries(qsf_lib PRIVATE m)
    target_link_libraries(qstream PRIVATE m)
endif()

# Windows-specific links
if(WIN32)
    target_link_libraries(qsf_lib PRIVATE advapi32)
endif()

# ── Optional LZ4 support ────────────────────────────────────────────
find_library(LZ4_LIB lz4)
find_path(LZ4_INCLUDE lz4.h)
if(LZ4_LIB AND LZ4_INCLUDE)
    message(STATUS "LZ4 found: ${LZ4_LIB}")
    target_compile_definitions(qsf_lib PRIVATE QSF_HAS_LZ4=1)
    target_include_directories(qsf_lib PRIVATE ${LZ4_INCLUDE})
    target_link_libraries(qsf_lib PRIVATE ${LZ4_LIB})
else()
    message(STATUS "LZ4 not found — compression support disabled")
endif()

# ── Install ──────────────────────────────────────────────────────────
install(TARGETS qstream DESTINATION bin)
install(TARGETS qsf_lib DESTINATION lib)
install(DIRECTORY include/qsf DESTINATION include)

# ── Summary ──────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "=== QStream Build Configuration ===")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler:      ${CMAKE_C_COMPILER_ID}")
message(STATUS "  Processor:     ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Platform:      ${CMAKE_SYSTEM_NAME}")
message(STATUS "===================================")
message(STATUS "")
