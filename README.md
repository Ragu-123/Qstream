# Qstream

Qstream is a C11 prototype implementation of the `QSF` (Quantized Streaming Format)
execution path from `immplementationplan.md`, focused on CPU-first, low-memory
inference building blocks that are more compact than GGUF-style full-tensor loading.

## What is implemented

- **QSF v1 binary format primitives**
  - 128-byte fixed header
  - 32-byte per-layer index entries
  - CRC32 header integrity validation
- **Memory foundation**
  - 64-byte aligned arena allocator for predictable footprint
- **CPU dispatch foundation**
  - x86 SSE4.2 / AVX2 / AVX512 feature detection
  - ARM NEON compile-time detection
- **Fused scalar quantized kernels (MVP)**
  - 2-bit and 4-bit fused dequant + matvec (block size 64)
  - no full dequantized matrix materialization
- **CLI utility**
  - create a demo QSF file
  - inspect QSF metadata + layer index
  - run deterministic kernel checksum demos

## Why this is compact

Instead of dequantizing full layers into float32 buffers, the kernels dequantize each
quantized block directly in the hot loop and consume it immediately. This avoids
allocating large transient matrices and aligns with the plan's low-memory constraint.

## Build

```bash
make
```

## Usage

Create demo file:

```bash
./qstream demo-create demo.qsf
```

Inspect file:

```bash
./qstream inspect demo.qsf
```

Run kernel demos:

```bash
./qstream matvec-demo 2
./qstream matvec-demo 4
```

## Next planned increments

- SIMD kernels (AVX2/AVX512/NEON)
- mmap/pread double-buffered streaming
- real converter from HuggingFace safetensors into `.qsf`
- transformer layer execution loop with KV cache quantization
