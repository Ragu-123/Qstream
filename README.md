# Qstream

Qstream now implements a **single-phase integrated CPU pipeline** from the
`immplementationplan.md` direction: format parsing, memory primitives,
streaming layer I/O, quantized KV cache, core kernels, and an end-to-end demo
path in one runnable binary.

## Implemented in one phase

- QSF v1 header + layer index parsing and validation with CRC32.
- 64-byte aligned arena allocator.
- CPU feature detection (SSE4.2/AVX2/AVX512F/NEON).
- Layer streaming runtime (`open`, `index load`, double buffers, per-layer read).
- Quantized KV cache (2-bit / 4-bit store + dequantized retrieval).
- Fused dequant + matvec scalar kernels (2-bit and 4-bit, block size 64).
- Transformer utility kernels: RMSNorm, SiLU, vec add/mul/scale,
  top-k filtering, temperature softmax, argmax sampler.
- End-to-end `single-phase-demo` command wiring all components together.

## Build

```bash
make
```

## Usage

```bash
./qstream demo-create demo.qsf
./qstream inspect demo.qsf
./qstream single-phase-demo demo.qsf
```

The last command exercises all major subsystems in one execution path.
