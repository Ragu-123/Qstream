# Ultra-Low-Memory CPU Inference Engine: Complete Plan

## C vs C++ Decision

### Why Pure C Wins for This Project

```
Factor                  | C                        | C++
------------------------|--------------------------|---------------------------
Memory control          | You own every byte       | Hidden allocations everywhere
                        |                          | (vtables, RTTI, exceptions,
                        |                          | string copies, vector resizing)

Cache predictability    | Structs are flat, you     | Object layouts are compiler-
                        | know exact memory layout  | dependent, inheritance adds
                        |                          | indirection, cache misses

Compiler optimization   | Compiler sees everything, | Templates can bloat code,
                        | no hidden calls           | inlining unpredictable,
                        |                          | exception handling adds
                        |                          | invisible branches

SIMD intrinsics         | Works identically         | Works identically (tie)

Startup time            | No static constructors,   | Global objects, static init
                        | no runtime init           | order fiasco, slower start

Binary size             | Tiny (50-100KB)           | Larger (STL pulls in 500KB+)

ABI stability           | Stable across compilers   | Name mangling, ABI breaks

Debugging hot path      | What you write = what     | Need to mentally unfold
                        | executes                  | templates, operator overloads,
                        |                          | implicit conversions

Restrict keyword        | `restrict` fully          | No standard `restrict`,
                        | supported, enables        | compiler-specific __restrict
                        | critical alias analysis   |

Linking with anything   | Universal C ABI           | C++ ABI is a nightmare
```

### The Killer Reason: `restrict`

```
In our hot loop, the compiler MUST know that input and output
buffers don't overlap. C's `restrict` keyword enables this.
This alone can give 15-30% speedup in matrix operations because
the compiler can:
  - Reorder loads and stores freely
  - Vectorize more aggressively
  - Eliminate redundant loads
```

### When C++ Would Win (But We Don't Need It)

```
- Large team projects (namespaces, access control)
- Complex data structure libraries (templates)
- GUI applications
- When developer velocity matters more than runtime performance

None of these apply to a focused inference kernel.
```

### Verdict: **Pure C11** with:
- `_Alignas` for SIMD alignment
- `_Thread_local` for thread-local scratch buffers
- `_Static_assert` for compile-time checks
- `restrict` everywhere on hot paths
- `stdatomic.h` for lock-free prefetch signaling

---

# COMPLETE DETAILED PLAN

## Phase 0: Foundation Decisions

### 0.1 Target Platforms
```
Primary:
  - x86_64 with AVX2 (most modern PCs, 2013+)
  - x86_64 with AVX-512 (Intel 10th gen+, optional fast path)
  - ARM64 with NEON (Apple Silicon, Raspberry Pi 4/5, Android)

Secondary:
  - x86_64 with SSE4.2 only (fallback for old machines)
  - WASM SIMD (browser execution)

Detection:
  - Runtime CPUID check on x86
  - Compile-time #ifdef for ARM
  - Function pointers to select best kernel at startup
```

### 0.2 Memory Budget Accounting
```
Total budget: 200 MB (user configurable)

Breakdown strategy:
  ┌─────────────────────────────────────────┐
  │ Category           │ Budget │ Priority  │
  ├────────────────────┼────────┼───────────┤
  │ Layer buffer A     │ 40 MB  │ Essential │ ← currently computing
  │ Layer buffer B     │ 40 MB  │ Essential │ ← prefetching next
  │ KV cache           │ 60 MB  │ Essential │ ← compressed
  │ Activation buffers │ 20 MB  │ Essential │
  │ Embedding table    │ 30 MB  │ Essential │ ← partial/mmap
  │ Scratch space      │ 5 MB   │ Essential │
  │ Layer index + meta │ 1 MB   │ Essential │
  │ Thread stacks      │ 2 MB   │ Essential │
  │ Safety margin      │ 2 MB   │ Reserved  │
  └────────────────────┴────────┴───────────┘
  Total: 200 MB

  If model is small enough (GPT-2), some categories shrink
  and we gain headroom.

  If model is large (7B+), we tighten:
    - Smaller prefetch buffer
    - More aggressive KV quantization (2-bit instead of 4-bit)
    - Partial embedding loading
```

### 0.3 File Format Name: **"QSF" (Quantized Streaming Format)**
```
Principles:
  - Single file, everything included
  - Seekable: can jump to any layer instantly
  - Streamable: can read front-to-back sequentially
  - Self-describing: header contains all metadata
  - Checksummed: detect corruption
  - Extensible: version field + reserved bytes
```

---

## Phase 1: File Format Specification

### 1.1 File Layout (byte-level detail)
```
Offset 0x0000: FILE HEADER (128 bytes, fixed size)
  Bytes 0-3:     Magic number "QSF1" (0x51534631)
  Bytes 4-7:     Format version (uint32, currently 1)
  Bytes 8-11:    Header size (uint32, 128 for v1)
  Bytes 12-15:   Architecture type:
                   0 = GPT-2
                   1 = LLaMA
                   2 = Mistral
                   3 = Phi
                   4 = Custom (read arch config section)
  Bytes 16-19:   Number of transformer layers (uint32)
  Bytes 20-23:   Hidden dimension (uint32)
  Bytes 24-27:   Number of attention heads (uint32)
  Bytes 28-31:   Number of KV heads (for GQA, 0 = same as attn heads)
  Bytes 32-35:   Vocabulary size (uint32)
  Bytes 36-39:   Max sequence length (uint32)
  Bytes 40-43:   Intermediate dimension (FFN) (uint32)
  Bytes 44-47:   Head dimension (uint32, usually hidden/heads)
  Bytes 48-48:   Default quantization type:
                   0 = 2-bit symmetric
                   1 = 2-bit asymmetric
                   2 = 3-bit
                   3 = 4-bit
                   4 = 1.58-bit ternary
                   5 = Mixed (per-tensor, read tensor headers)
  Bytes 49-49:   Activation type:
                   0 = GELU
                   1 = SiLU/Swish
                   2 = ReLU
  Bytes 50-50:   Normalization type:
                   0 = LayerNorm
                   1 = RMSNorm
  Bytes 51-51:   Position encoding type:
                   0 = Learned
                   1 = RoPE
                   2 = ALiBi
  Bytes 52-55:   RoPE theta (float32, if RoPE)
  Bytes 56-63:   Layer index offset (uint64, byte offset in file)
  Bytes 64-71:   Embedding section offset (uint64)
  Bytes 72-79:   Final norm + output head offset (uint64)
  Bytes 80-83:   BOS token ID (uint32)
  Bytes 84-87:   EOS token ID (uint32)
  Bytes 88-91:   PAD token ID (uint32)
  Bytes 92-95:   Total file size (uint32, for integrity check, lower 32 bits)
  Bytes 96-99:   CRC32 of header bytes 0-95
  Bytes 100-127: Reserved (zeros)

Offset LAYER_INDEX_OFFSET: LAYER INDEX TABLE
  Per layer (32 bytes each):
    Bytes 0-7:   Layer data offset in file (uint64)
    Bytes 8-11:  Compressed size on disk (uint32)
    Bytes 12-15: Decompressed size (uint32) - after LZ4, before dequant
    Bytes 16-16: Quantization type for this layer (uint8)
    Bytes 17-17: Compression type:
                   0 = None (raw quantized)
                   1 = LZ4
                   2 = LZ4-HC (higher compression)
                   3 = ZSTD (if size matters more than speed)
    Bytes 18-19: Number of tensors in this layer (uint16)
    Bytes 20-23: CRC32 of compressed data
    Bytes 24-27: Importance score (float32, for layer skipping)
    Bytes 28-31: Reserved

  Total index size: num_layers * 32 bytes
  Example: 32 layers = 1024 bytes. Trivial.

EMBEDDING SECTION (at embedding offset):
  Embedding header (16 bytes):
    Bytes 0-3:   Quant type (uint32)
    Bytes 4-7:   Compressed size (uint32)
    Bytes 8-11:  Number of embedding vectors (= vocab_size)
    Bytes 12-15: Embedding dimension

  Followed by:
    Quantized embedding data
    If vocabulary > 50K: split into chunks for partial loading

LAYER DATA SECTIONS (one per layer):
  Each layer contains multiple tensors, laid out as:

  TENSOR HEADER (16 bytes per tensor):
    Bytes 0-3:   Tensor type:
                   0 = attention Q weight
                   1 = attention K weight
                   2 = attention V weight
                   3 = attention output weight
                   4 = FFN gate weight (for SiLU architectures)
                   5 = FFN up weight
                   6 = FFN down weight
                   7 = attention norm weight
                   8 = FFN norm weight
                   9 = attention Q bias
                   10 = attention K bias
                   11 = attention V bias
                   12 = attention output bias
                   13 = FFN biases
    Bytes 4-7:   Rows (uint32)
    Bytes 8-11:  Cols (uint32)
    Bytes 12-12: Tensor-specific quant type (uint8, 0xFF = use layer default)
    Bytes 13-15: Reserved

  TENSOR DATA (after header):
    Quantization block data
    Each block: [scale (fp16)] [zero_point/min (fp16)] [packed_bits]
    Block size: 64 values (configurable)

    Quantization block layout for 2-bit:
      2 bytes: scale (float16)
      2 bytes: min/zero (float16)
      16 bytes: 64 values * 2 bits = 128 bits = 16 bytes
      Total: 20 bytes per block = 2.5 bits/value effective

    Quantization block layout for ternary (1.58-bit):
      2 bytes: scale (float16)
      8 bytes: sign bits (64 bits)
      8 bytes: zero mask (64 bits)
      Total: 18 bytes per block = 2.25 bits/value effective

    Quantization block layout for 4-bit:
      2 bytes: scale (float16)
      2 bytes: zero point (float16)
      32 bytes: 64 values * 4 bits = 256 bits = 32 bytes
      Total: 36 bytes per block = 4.5 bits/value effective

FINAL SECTION (at final offset):
  Final layer norm weights (float16, hidden_dim * 2 bytes)
  Final layer norm bias (float16, hidden_dim * 2 bytes, if applicable)
  Output projection / LM head:
    If tied with embeddings: flag byte = 0xFF, use embedding table
    If separate: quantized weight matrix [vocab_size x hidden_dim]
```

### 1.2 Conversion Tool Plan
```
Input: HuggingFace model directory (pytorch_model.bin or safetensors)
Output: .qsf file

Steps:
  1. Load model config.json → extract architecture params
  2. Load tokenizer info → extract special token IDs
  3. For each parameter tensor:
     a. Read as float32
     b. Compute importance scores:
        - Per-row L2 norm
        - Per-row activation variance (if calibration data available)
     c. Choose quantization strategy:
        - Attention QKV: 2-bit or ternary (these are most compressible)
        - FFN up/gate: 2-bit
        - FFN down: 4-bit (more sensitive)
        - Norms: float16 (tiny, keep precise)
        - Embeddings: 4-bit (need more precision for rare tokens)
     d. Quantize using chosen strategy
     e. Pack into blocks
  4. Optionally compress each layer with LZ4
  5. Compute layer importance scores:
     - Run calibration prompts
     - Measure output change when layer is zeroed out
     - Store as float32 in layer index
  6. Write file in specified layout
  7. Compute and write all CRC32 checksums

Calibration (optional but recommended):
  - User provides ~100 sample prompts
  - Tool runs full-precision forward pass
  - Measures per-layer and per-tensor sensitivity
  - Adjusts bit-widths to minimize quality loss within size budget
```

---

## Phase 2: Memory Management System

### 2.1 Arena Allocator
```
Why: malloc/free is slow and fragments memory.
We know our allocation pattern: allocate at startup, never free until shutdown.

Design:
  - Single large allocation at startup (200MB or user-specified)
  - Carve out sub-regions with simple bump pointer
  - Each sub-region has a name (for debugging)
  - Alignment guaranteed to 64 bytes (cache line + AVX-512)

  Arena structure:
    base_ptr:      pointer to start of allocated region
    current_ptr:   next available position
    end_ptr:       hard limit
    total_size:    total arena size
    used_size:     current usage
    allocations[]: array of {name, ptr, size} for debugging

  Operations:
    arena_create(size_mb) → Arena*
    arena_alloc(arena, size, alignment, name) → void*
    arena_reset(arena)  — reset to empty (for scratch buffers)
    arena_stats(arena)  — print usage report
    arena_destroy(arena)

  Sub-arenas for different purposes:
    persistent_arena: things that live forever (layer index, config)
    layer_arena: double-buffered layer data (reset per layer)
    scratch_arena: temporary computation buffers (reset per operation)
    kv_arena: KV cache (grows during generation)
```

### 2.2 Memory-Mapped File Access
```
For the model file:
  - mmap() the entire file as read-only
  - madvise(MADV_SEQUENTIAL) for sequential layer reading
  - madvise(MADV_WILLNEED) on next layer (prefetch hint)
  - madvise(MADV_DONTNEED) on previous layer (release hint)

Alternative for systems without good mmap:
  - Use pread() with explicit double buffer
  - Better control over when I/O happens

Decision: Support BOTH, choose at runtime:
  - If file is on SSD: mmap is fine
  - If file is on HDD or network: explicit pread with large reads

Platform abstraction:
  - Linux: mmap, madvise, pread
  - macOS: mmap, madvise, pread (same API)
  - Windows: CreateFileMapping, MapViewOfFile, ReadFile
  - Wrapper functions hide platform differences
```

### 2.3 Allocation Map for 200MB Budget
```
This is computed at startup based on model size:

  budget_compute(model_params):
    layer_compressed_max = max(layer_index[i].compressed_size for all i)
    layer_decompressed_max = max(layer_index[i].decompressed_size for all i)

    // Double buffer for layers
    layer_buf_a = layer_compressed_max (rounded up to page size)
    layer_buf_b = layer_compressed_max

    // Decompression target (LZ4 output)
    decompress_buf = layer_decompressed_max

    // Activation buffers
    // For single token generation: hidden_dim * sizeof(float)
    // For prompt processing: max_batch * hidden_dim * sizeof(float)
    activation_buf = max_seq_len * hidden_dim * 4  // float32

    // Scratch for matmul intermediate
    scratch = max(hidden_dim, intermediate_dim) * 4  // float32

    // KV cache
    kv_per_layer = 2 * num_kv_heads * head_dim * max_seq_len * 0.5  // 4-bit = 0.5 bytes
    kv_total = kv_per_layer * num_layers

    // Embedding lookup cache
    // Don't load entire embedding table
    // Cache recently used embeddings
    embedding_cache = 1000 * hidden_dim * 2  // cache 1000 embeddings in fp16

    // Norm weights (always resident, tiny)
    norm_weights = num_layers * hidden_dim * 2 * 2  // 2 norms per layer, fp16

    total = layer_buf_a + layer_buf_b + decompress_buf +
            activation_buf + scratch + kv_total +
            embedding_cache + norm_weights

    if total > budget:
      // Reduce in priority order:
      1. Reduce max_seq_len (shrinks KV cache and activations)
      2. Use 2-bit KV cache instead of 4-bit
      3. Reduce embedding cache size
      4. Eliminate double buffer (slower, no prefetch)
      5. Error: model too large for budget

    print memory map for user
```

---

## Phase 3: Quantization Engine

### 3.1 Offline Quantization (Conversion Time)
```
For each weight tensor:

  Step 1: Statistical analysis
    - Compute per-channel (per-row) statistics:
      mean, std, min, max, L2 norm
    - Compute per-block statistics (block = 64 elements):
      min, max, range
    - Detect outliers:
      values > 3 standard deviations from mean

  Step 2: Outlier handling
    - If outlier percentage < 0.1%:
      Clamp outliers (simple, minimal quality loss)
    - If outlier percentage 0.1-1%:
      Use mixed precision: outliers stored separately in fp16
      Main data in low-bit
    - If outlier percentage > 1%:
      Bump up to 4-bit for this tensor

  Step 3: Block quantization
    For each block of 64 values:
      a. Find block min and max
      b. Compute scale = (max - min) / (2^bits - 1)
      c. Quantize: q = round((value - min) / scale)
      d. Clamp q to [0, 2^bits - 1]
      e. Pack bits tightly
      f. Store scale and min as float16

  Step 4: Quality verification (if calibration data available)
    - Dequantize and compute MSE vs original
    - If MSE > threshold: bump bit-width for this tensor
    - Compute cosine similarity of dequantized vs original
    - Target: cosine similarity > 0.99

  Ternary quantization (BitNet-style):
    For each block of 64 values:
      a. Compute mean absolute value → scale
      b. For each value:
         if |value| < 0.3 * scale: quantize to 0
         else if value > 0: quantize to +1
         else: quantize to -1
      c. Store as sign_bits (64 bits) + zero_mask (64 bits) + scale (fp16)
```

### 3.2 Runtime Dequantization (Fused with Compute)
```
CRITICAL DESIGN DECISION:
  We NEVER fully dequantize a tensor into a float buffer.
  Dequantization happens inside the matmul kernel,
  one block at a time, values go directly into FMA.

Why this matters:
  - A 4096x4096 weight matrix dequantized = 64MB (float32)
  - We literally cannot afford this in 200MB budget
  - Fused approach: dequantize 64 values → use immediately → discard
  - Memory used for dequantized values: 256 bytes (one block in registers)

The fused kernel reads quantized blocks, converts to float in registers,
multiplies with input, accumulates, and moves on. No intermediate buffer.
```

### 3.3 Quantization Block Sizes
```
Why 64 values per block?
  - AVX-512 processes 16 floats at once
  - 64 = 4 * 16 → exactly 4 AVX-512 iterations per block
  - NEON processes 4 floats at once
  - 64 = 16 * 4 → exactly 16 NEON iterations per block
  - Good balance between overhead (scale/min per block) and accuracy
  - Smaller blocks = better accuracy but more overhead
  - Larger blocks = worse accuracy but less overhead

Alternative: 128 or 256 for very large models (saves header space)
Make this configurable in format, stored in header.
```

---

## Phase 4: Compute Kernels

### 4.1 Kernel Selection Architecture
```
At startup:
  1. Detect CPU features (CPUID on x86, /proc/cpuinfo on ARM)
  2. Set function pointers to best available kernel

  kernel_table = {
    .matvec_2bit = NULL,
    .matvec_4bit = NULL,
    .matvec_ternary = NULL,
    .layer_norm = NULL,
    .rms_norm = NULL,
    .softmax = NULL,
    .rope = NULL,
    .silu = NULL,
    .gelu = NULL,
    .embedding_lookup = NULL,
    .top_k_sample = NULL,
  }

  if (has_avx512):
    kernel_table.matvec_2bit = fused_2bit_matvec_avx512
    kernel_table.matvec_4bit = fused_4bit_matvec_avx512
    // ... etc
  elif (has_avx2):
    kernel_table.matvec_2bit = fused_2bit_matvec_avx2
    // ... etc
  elif (has_neon):
    kernel_table.matvec_2bit = fused_2bit_matvec_neon
    // ... etc
  else:
    kernel_table.matvec_2bit = fused_2bit_matvec_scalar
    // ... etc
```

### 4.2 Kernel List (every single kernel needed)
```
Core math kernels:
  1. fused_dequant_matvec_2bit(weights_q, input, output, rows, cols)
     Matrix-vector multiply with on-the-fly 2-bit dequantization
     Variants: avx512, avx2, neon, scalar

  2. fused_dequant_matvec_4bit(...)
     Same for 4-bit

  3. fused_dequant_matvec_ternary(...)
     Ternary: no multiply needed, just add/subtract/skip

  4. fused_dequant_matmul_2bit(weights_q, input_matrix, output, M, N, K)
     Matrix-matrix multiply for prompt processing (multiple tokens at once)
     input is [M x K] float, weights are [N x K] quantized
     output is [M x N] float

  5. fused_dequant_matmul_4bit(...)
     Same for 4-bit

Attention kernels:
  6. attention_qkv_project(input, q_weight, k_weight, v_weight,
                           q_out, k_out, v_out,
                           hidden_dim, num_heads, head_dim)
     Can fuse all three projections for better cache usage
     Reads input once, produces Q, K, V

  7. attention_scores(query, key_cache, scores,
                      head_dim, seq_len, num_heads)
     Computes Q * K^T / sqrt(head_dim)
     Key cache is quantized → dequantize during dot product

  8. attention_softmax_inplace(scores, seq_len, num_heads)
     Numerically stable softmax with online max tracking

  9. attention_weighted_sum(scores, value_cache, output,
                           seq_len, num_heads, head_dim)
     Computes softmax_scores * V
     Value cache is quantized → dequantize during multiply

  10. causal_mask_apply(scores, current_pos, seq_len)
      Set future positions to -infinity

Normalization kernels:
  11. layer_norm(input, output, weight, bias, dim, epsilon)
      Standard LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta

  12. rms_norm(input, output, weight, dim, epsilon)
      RMSNorm: x / sqrt(mean(x^2) + eps) * gamma
      Used by LLaMA, Mistral

Position encoding kernels:
  13. rope_apply(q, k, head_dim, position, theta)
      Rotary position embeddings
      Apply rotation to Q and K vectors based on position

  14. rope_apply_batched(q, k, head_dim, positions[], num_positions, theta)
      Batched version for prompt processing

Activation kernels:
  15. silu_inplace(x, size)
      SiLU: x * sigmoid(x)
      Used by LLaMA, Mistral

  16. gelu_inplace(x, size)
      GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
      Used by GPT-2

  17. gelu_fast_inplace(x, size)
      Approximate GELU using lookup table + interpolation

Element-wise kernels:
  18. vec_add(a, b, out, size)
      Residual connections

  19. vec_mul(a, b, out, size)
      Element-wise multiply (for gated FFN: gate * up)

  20. vec_scale(x, scale, size)
      Multiply all elements by scalar

Sampling kernels:
  21. softmax_with_temperature(logits, probs, vocab_size, temperature)

  22. top_k_filter(logits, vocab_size, k)
      Zero out everything except top-k logits

  23. top_p_filter(logits, vocab_size, p)
      Nucleus sampling: keep tokens until cumulative prob > p

  24. sample_from_probs(probs, vocab_size, rng_state) → token_id

Utility kernels:
  25. fp16_to_fp32_array(input_fp16, output_fp32, count)
      Bulk convert norm weights from stored fp16 to working fp32

  26. quantize_kv_4bit(float_input, quantized_output, scale_out, dim)
      Quantize KV values during generation for compressed cache

  27. dequantize_kv_4bit(quantized_input, scale, float_output, dim)
      Dequantize KV during attention computation

  28. memset_pattern(dst, pattern, size)
      Fast memory initialization with SIMD
```

### 4.3 Kernel Optimization Details
```
For every kernel above, these optimizations apply:

  Cache optimization:
    - Process data in L1-cache-sized chunks (32-48KB)
    - Prefetch next chunk while processing current:
      _mm_prefetch(ptr + 512, _MM_HINT_T0)  // prefetch into L1
      _mm_prefetch(ptr + 2048, _MM_HINT_T1) // prefetch into L2
    - Ensure input and output don't alias (restrict keyword)

  Register pressure management:
    - Use 4 independent accumulators to hide FMA latency
    - FMA latency is 4-5 cycles, throughput is 0.5 cycles
    - Need 8-10 independent operations in flight
    - 4 accumulators * 2 operations each (load + FMA) ≈ 8

  Loop structure:
    - Main loop: processes 4 blocks per iteration (256 elements)
    - Cleanup loop: handles remainder (< 256 elements)
    - No branches in main loop body

  Alignment:
    - All buffers aligned to 64 bytes
    - Use aligned loads (_mm512_load_ps) in main loop
    - Use unaligned loads only in cleanup

  Threading:
    - Large matrices: split rows across threads
    - Use thread pool (not pthread_create per operation!)
    - Each thread gets contiguous rows for cache locality
    - Minimum chunk: 64 rows per thread (avoid false sharing)
```

---

## Phase 5: Transformer Execution Pipeline

### 5.1 Single Layer Execution (Token Generation Mode)
```
Input: activation vector [hidden_dim] (float32)
Output: updated activation vector [hidden_dim] (float32)
Side effect: updates KV cache for this layer

Step-by-step:

  5.1.1 ATTENTION BLOCK:

    a) Pre-attention norm:
       If LayerNorm:
         residual = copy of input
         normed = LayerNorm(input, attn_norm_weight, attn_norm_bias)
       If RMSNorm:
         residual = copy of input
         normed = RMSNorm(input, attn_norm_weight)

    b) QKV projection:
       Q = fused_dequant_matvec(Wq, normed)  // [num_heads * head_dim]
       K = fused_dequant_matvec(Wk, normed)  // [num_kv_heads * head_dim]
       V = fused_dequant_matvec(Wv, normed)  // [num_kv_heads * head_dim]

       If biases exist:
         Q += bq, K += bk, V += bv

    c) RoPE (if applicable):
       Apply rotary embeddings to Q and K using current position
       Operates on pairs of elements within each head

    d) KV cache update:
       Quantize K to 4-bit → store in kv_cache.k[layer][position]
       Quantize V to 4-bit → store in kv_cache.v[layer][position]

    e) Attention computation (per head):
       For each attention head h:
         q_h = Q[h * head_dim : (h+1) * head_dim]

         // Handle GQA: multiple Q heads share same KV head
         kv_h = h / (num_heads / num_kv_heads)

         // Compute attention scores
         For each cached position p from 0 to current_pos:
           k_p = dequantize_kv(kv_cache.k[layer][kv_h][p])
           score[p] = dot_product(q_h, k_p) / sqrt(head_dim)

         // Apply causal mask (all valid for generation, all past positions)
         // Apply softmax
         softmax_inplace(score, current_pos + 1)

         // Weighted sum of values
         out_h = zeros[head_dim]
         For each position p:
           v_p = dequantize_kv(kv_cache.v[layer][kv_h][p])
           out_h += score[p] * v_p

    f) Output projection:
       attn_output = fused_dequant_matvec(Wo, concat(out_h for all h))
       If bias: attn_output += bo

    g) Residual connection:
       hidden = residual + attn_output

  5.1.2 FFN BLOCK:

    a) Pre-FFN norm:
       residual = copy of hidden
       normed = Norm(hidden, ffn_norm_weight, ffn_norm_bias)

    b) FFN computation:
       If gated (LLaMA/Mistral):
         gate = fused_dequant_matvec(W_gate, normed)  // [intermediate_dim]
         up = fused_dequant_matvec(W_up, normed)      // [intermediate_dim]
         gate = silu(gate)
         intermediate = gate * up  // element-wise
       If standard (GPT-2):
         intermediate = fused_dequant_matvec(W_up, normed)
         intermediate = gelu(intermediate)

    c) Down projection:
       ffn_output = fused_dequant_matvec(W_down, intermediate)
       If bias: ffn_output += b_down

    d) Residual connection:
       hidden = residual + ffn_output

  5.1.3 Output:
    Return hidden (updated activation vector)
```

### 5.2 Prompt Processing Mode (Multiple Tokens)
```
Different from token generation:
  - Multiple tokens processed at once → matrix-matrix multiply
  - KV cache populated for all positions at once
  - Can parallelize across tokens

Key difference in computation:
  - Input is [num_tokens x hidden_dim] instead of [hidden_dim]
  - Use matmul kernels instead of matvec kernels
  - Attention becomes [num_tokens x num_tokens] score matrix
  - Much more compute, but builds full KV cache for fast generation

Optimization:
  - If num_tokens > some threshold (e.g., 32), process in chunks
  - Each chunk fits in cache
  - Overlap chunks with layer prefetching
```

### 5.3 Full Inference Flow
```
generate_token(engine, input_tokens, num_input_tokens):

  Phase A: Prompt Processing
    1. Embed all input tokens (batched lookup from quantized embedding table)
    2. For each layer L from 0 to num_layers-1:
       a. Prefetch layer L+1 data from disk (async)
       b. Decompress layer L (LZ4)
       c. Execute layer L on all input tokens (matmul mode)
       d. Release layer L data
       e. Wait for layer L+1 prefetch to complete
    3. Apply final norm
    4. Project to vocabulary (get logits for last token only)
    5. Sample next token

  Phase B: Token Generation (loop)
    6. While not EOS and not max_length:
       a. Embed the latest generated token
       b. For each layer L from 0 to num_layers-1:
          - Prefetch layer L+1
          - Decompress layer L
          - Execute layer L on single token (matvec mode)
          - Release layer L
       c. Apply final norm
       d. Project to vocabulary
       e. Sample next token
       f. Yield/output the token
       g. Advance position counter
```

---

## Phase 6: Layer Streaming System

### 6.1 Double Buffer Design
```
Two buffers: A and B
State machine:

  IDLE:
    Both buffers empty
    → Load layer 0 into A (synchronous, must wait)
    → Transition to EXECUTING_A

  EXECUTING_A:
    Buffer A: contains current layer, being used for computation
    Buffer B: FREE
    → Start async load of next layer into B
    → When computation finishes:
       → Wait for B to be ready (should already be done if disk is fast)
       → Transition to EXECUTING_B, mark A as FREE

  EXECUTING_B:
    Buffer B: contains current layer, being used for computation
    Buffer A: FREE
    → Start async load of next layer into A
    → When computation finishes:
       → Wait for A to be ready
       → Transition to EXECUTING_A, mark B as FREE

  This ping-pong continues for all layers.
```

### 6.2 Async I/O Implementation
```
Option 1: pthread + pread
  - Dedicated I/O thread
  - Signals completion via atomic flag
  - Compute thread checks flag, if not ready → spin briefly, then yield

Option 2: io_uring (Linux 5.1+)
  - Zero-copy, kernel-managed async I/O
  - Highest performance on Linux
  - Fallback to pthread for older kernels

Option 3: Overlapped I/O (Windows)
  - ReadFileEx with completion callback

Implementation:
  Use abstract interface:
    async_read_start(fd, buffer, offset, size) → handle
    async_read_poll(handle) → READY / NOT_READY
    async_read_wait(handle) → blocks until done
```

### 6.3 LZ4 Decompression Stage
```
After I/O completes, before computation:
  1. LZ4_decompress_safe(compressed_buf, decompressed_buf, 
                          compressed_size, max_decompressed_size)
  2. This gives us the raw quantized tensor data
  3. Parse tensor headers within the decompressed data
  4. Set up pointers to each tensor's quantized data
  5. These pointers are passed to fused dequant+compute kernels

LZ4 decompression speed: ~4-5 GB/s on modern CPU
For a 4MB compressed layer: ~1ms decompression time
This is negligible compared to matmul time.

Why LZ4 and not ZSTD:
  - LZ4 decompression is 3-5x faster than ZSTD
  - We decompress every layer on every token → speed matters
  - ZSTD better compression but not worth the CPU cost at runtime
  - LZ4 works well on quantized data (repetitive bit patterns)
  - Offer ZSTD as conversion-time option for smaller files,
    with runtime LZ4 re-compression
```

---

## Phase 7: KV Cache Management

### 7.1 Structure
```
KV cache layout in memory:

  For each layer:
    For each KV head:
      keys:   [max_seq_len x head_dim] quantized to 4-bit
      values: [max_seq_len x head_dim] quantized to 4-bit
      scales: [max_seq_len] float32 (one scale per position per head)

  Memory calculation for LLaMA-7B (32 layers, 32 heads, 128 head_dim, 2048 seq):
    Per position per head: 128 * 0.5 bytes = 64 bytes (4-bit)
    Per position: 32 heads * 64 bytes * 2 (K+V) = 4096 bytes
    Per layer: 2048 * 4096 = 8 MB
    All layers: 32 * 8 MB = 256 MB  ← TOO MUCH

  Solution: KV cache also needs to be budgeted!

  Options:
    a) 2-bit KV cache (halves the size, some quality loss)
    b) Sliding window attention (only cache last N positions)
       Mistral uses window=4096, we can use window=512-1024
    c) KV cache compression with more aggressive quantization:
       - Group 4 positions together, shared scale
       - Reduces scale storage overhead
    d) H2O (Heavy-Hitter Oracle): only keep important KV entries
       - During attention, track which positions have high scores
       - Evict low-scoring positions
       - Keep top-K most attended positions
       - Much smaller cache for long sequences

  Recommended: Sliding window (1024) + 4-bit quantization
    Per layer: 1024 * 4096 = 4 MB
    All layers: 32 * 4 = 128 MB... still tight

  More aggressive: Sliding window (512) + 2-bit KV
    Per layer: 512 * 32 * 128 * 0.25 * 2 = 1 MB
    All layers: 32 * 1 = 32 MB ← fits easily!
```

### 7.2 KV Cache Operations
```
  store_kv(cache, layer, position, key_float, value_float):
    1. Compute scale for this key vector: max(|key|) / max_quant_val
    2. Quantize key to 4-bit (or 2-bit)
    3. Store quantized key at cache.k[layer][position]
    4. Store scale at cache.k_scales[layer][position]
    5. Same for value

  retrieve_k(cache, layer, head, position) → float[head_dim]:
    1. Read quantized data from cache.k[layer][head][position]
    2. Read scale from cache.k_scales[layer][head][position]
    3. Dequantize to float
    4. Return
    (In practice, this happens inside attention kernel, not separately)

  evict(cache, layer, position):
    Mark position as invalid (for H2O eviction strategy)

  rotate_window(cache, window_size):
    When position > window_size:
      Shift all entries: position p becomes p - 1
      Evict position 0
      (Or: use circular buffer with head/tail pointers, O(1) eviction)
```

---

## Phase 8: Embedding Table Handling

### 8.1 Problem
```
Embedding tables are large:
  GPT-2 (50257 vocab, 768 hidden): 50257 * 768 * 4 = 154 MB (fp32)
  LLaMA (32000 vocab, 4096 hidden): 32000 * 4096 * 4 = 500 MB (fp32)

Even quantized:
  GPT-2 4-bit: ~19 MB
  LLaMA 4-bit: ~62 MB

For small models: load entire quantized embedding table into RAM.
For large models: need partial loading.
```

### 8.2 Solutions
```
Strategy A: Full quantized embedding (models ≤ 1B params)
  - Quantize entire embedding table to 4-bit or 8-bit
  - Load into RAM at startup
  - Lookup is just: read quantized row, dequantize to float
  - Size: 19-62 MB depending on model

Strategy B: LRU embedding cache (models > 1B params)
  - Memory-map the embedding section of the file
  - Keep LRU cache of recently used embeddings in RAM
  - Cache size: 2000 embeddings * hidden_dim * 2 bytes (fp16) 
    = 2000 * 4096 * 2 = 16 MB
  - Most text uses ~2000-5000 unique tokens
  - Cache hit rate in practice: 90%+ after first prompt
  - Cache miss: read from mmap'd file, dequantize, insert into cache

Strategy C: Clustered embedding (most aggressive)
  - Offline: cluster vocabulary into 256 clusters (k-means)
  - Store cluster centroids [256 x hidden_dim] in RAM (8 MB for 4096 dim)
  - Store per-token residual from centroid [vocab_size x hidden_dim] 
    quantized to 2-bit
  - Lookup: centroid[cluster_id] + dequantize(residual[token_id])
  - Much smaller on disk AND in RAM
```

---

## Phase 9: Output/LM Head Handling

### 9.1 Problem
```
Output projection: hidden_dim → vocab_size (weight tying with embeddings or separate)
This is a huge matrix: [vocab_size x hidden_dim]
Same size as embedding table.

We only need the TOP-K logits, not all 50K+ logits.
```

### 9.2 Solution: Approximate Top-K Output
```
Strategy: Don't compute all vocab_size dot products!

  Step 1: Cluster output vectors offline (same 256 clusters as embeddings)

  Step 2: At inference time:
    a. Compute dot product of hidden_state with each cluster centroid (256 dot products)
    b. Select top-8 clusters (highest centroid dot products)
    c. Compute exact dot products only for tokens in those 8 clusters
       (8 clusters × ~200 tokens per cluster = ~1600 exact computations)
    d. Return top-K from this reduced set

  Accuracy: >99% of the time, the correct top-1 token is in the top-8 clusters
  Speedup: ~30x compared to computing all vocab_size dot products

  Alternative: If using weight tying, the embedding cache already has the vectors
  we need. Iterate over cached embeddings + cluster approach for uncached.
```

---

## Phase 10: Early Exit / Layer Skipping

### 10.1 Importance-Based Layer Skipping
```
During conversion:
  - Run calibration data through full model
  - For each layer, measure:
    a. Cosine similarity between input and output of that layer
       High similarity = layer doesn't change much = skippable
    b. Effect on final output when layer is skipped
       Small effect = safe to skip

  Store importance score (float32) in layer index

At runtime:
  - User sets speed/quality tradeoff parameter (0.0 = full quality, 1.0 = max speed)
  - Skip layers with importance below threshold
  - Always keep first 2 and last 2 layers (most important empirically)
  - Skip probability increases for middle layers

  Expected savings: 20-40% of layers can be skipped with <5% quality loss
```

### 10.2 Confidence-Based Early Exit
```
Periodically (every N layers), check if output is already confident:

  1. Apply final norm to current hidden state
  2. Compute dot product with top-100 most common token embeddings
     (these 100 embeddings are always in RAM, ~3KB)
  3. Compute entropy of resulting distribution
  4. If entropy < threshold → exit early

This costs ~0.1ms per check. If it saves skipping 10 layers (each ~5ms), 
net saving is huge.

Train a tiny classifier (hidden_dim → 1 sigmoid) per-layer to predict
whether early exit is safe. This classifier is <1KB per layer.
```

---

## Phase 11: Threading Model

### 11.1 Thread Pool
```
Create thread pool at startup:
  - Number of threads = number of physical cores (not hyperthreads)
  - Each thread pinned to a core (set affinity)
  - Threads wait on condition variable for work
  - Work items are (function_ptr, arg_ptr, start_row, end_row)

Thread pool operations:
  pool_create(num_threads) → ThreadPool*
  pool_submit(pool, func, arg, num_items) — divide items across threads
  pool_wait(pool) — wait for all threads to finish
  pool_destroy(pool)

Where threading is used:
  - Matrix-vector multiply: split rows across threads
    (each thread computes a subset of output rows)
  - Prompt processing matrix multiply: split across both dimensions
  - Attention score computation: split across heads
  - Softmax: each thread handles a subset of heads

Where threading is NOT used (too little work):
  - Layer norm / RMS norm (sequential, ~1 microsecond)
  - Activation functions on single vector
  - Embedding lookup (single row read)
  - KV cache store (single position)
```

### 11.2 Thread for I/O
```
Separate from compute thread pool:
  - One dedicated I/O thread
  - Only does: read from file, LZ4 decompress
  - Signals completion via atomic flag
  - Never does compute (avoid interfering with compute threads)

Communication with main thread:
  Lockless single-producer single-consumer queue
  Or simple atomic flag:
    io_thread sets: layer_ready[buffer_id] = 1
    main thread polls: while (!layer_ready[buffer_id]) { _mm_pause(); }
```

---

## Phase 12: Tokenizer

### 12.1 Requirements
```
Need a tokenizer to convert text → token IDs and back.

Options:
  a) BPE tokenizer (GPT-2, LLaMA)
  b) SentencePiece (LLaMA)
  c) WordPiece (BERT, not relevant here)

Implementation:
  - Store tokenizer vocabulary + merge rules in a separate file
    or embed in the .qsf file as a section
  - Implement BPE encoding/decoding in C
  - This is not performance-critical (runs once per prompt)
  - Can be simple and correct, doesn't need to be fast

Data structures:
  - Token ID → string: simple array of pointers
  - String → token ID: hash table
  - Merge rules: ordered list of (pair → merged) rules

Encoding algorithm:
  1. Split text into characters (UTF-8 aware)
  2. Convert characters to initial token IDs
  3. Repeatedly find highest-priority merge pair and apply
  4. Continue until no more merges apply
  5. Return list of token IDs

Decoding algorithm:
  1. For each token ID, look up string
  2. Concatenate
  3. Handle special characters (Ġ for space in GPT-2, ▁ for SentencePiece)
```

---

## Phase 13: Platform Abstraction Layer

### 13.1 What Needs Abstraction
```
File I/O:
  - open, read, pread, close
  - mmap, munmap, madvise
  - Windows equivalents

Memory:
  - aligned_alloc / _aligned_malloc
  - mmap for large allocations
  - huge pages (optional, 2MB/1GB pages for large allocations)

Threading:
  - pthread / Windows threads
  - Atomic operations
  - Thread affinity setting

CPU Detection:
  - CPUID instruction (x86)
  - getauxval/AT_HWCAP (ARM Linux)
  - sysctl (macOS ARM)

Timer:
  - clock_gettime / QueryPerformanceCounter
  - For benchmarking and profiling

Console:
  - Streaming output (printing tokens as they're generated)
  - UTF-8 output support
  - Windows console mode setting for UTF-8
```

---

## Phase 14: User Interface / API

### 14.1 C API
```
// Public API (qsf.h)

typedef struct QSFEngine QSFEngine;
typedef struct QSFConfig QSFConfig;

// Configuration
QSFConfig* qsf_config_default(void);
void qsf_config_set_ram_budget(QSFConfig*, size_t megabytes);
void qsf_config_set_threads(QSFConfig*, int num_threads);
void qsf_config_set_max_seq_len(QSFConfig*, int max_seq_len);
void qsf_config_set_quality(QSFConfig*, float quality); // 0.0-1.0
void qsf_config_free(QSFConfig*);

// Engine lifecycle
QSFEngine* qsf_load(const char* model_path, QSFConfig* config);
void qsf_free(QSFEngine*);

// Generation
typedef void (*QSFTokenCallback)(int token_id, const char* text, void* user_data);

void qsf_generate(QSFEngine*, const char* prompt,
                   int max_tokens, float temperature, float top_p,
                   QSFTokenCallback callback, void* user_data);

// Lower-level API
int* qsf_tokenize(QSFEngine*, const char* text, int* num_tokens);
char* qsf_detokenize(QSFEngine*, const int* tokens, int num_tokens);
float* qsf_forward(QSFEngine*, const int* tokens, int num_tokens);
int qsf_sample(const float* logits, int vocab_size, 
                float temperature, float top_p);

// Stats
void qsf_print_stats(QSFEngine*);  // RAM usage, tokens/sec, etc.
```

### 14.2 CLI Tool
```
Usage: qsf-run [options] <model.qsf>

Options:
  --prompt "text"        Input prompt
  --max-tokens N         Maximum tokens to generate (default: 256)
  --temperature F        Sampling temperature (default: 0.7)
  --top-p F             Top-p sampling (default: 0.9)
  --top-k N             Top-k sampling (default: 40)
  --ram-budget N         RAM budget in MB (default: 200)
  --threads N            Number of threads (default: auto)
  --quality F            Quality/speed tradeoff 0-1 (default: 1.0)
  --interactive          Interactive chat mode
  --bench                Run benchmark and exit
  --verbose              Print memory layout, kernel selection, etc.

Conversion tool:
  qsf-convert [options] <input_model_dir> <output.qsf>

  --quant-type TYPE      Quantization type: 2bit/3bit/4bit/ternary/mixed
  --calibration FILE     Calibration data for optimal quantization
  --compress TYPE        Compression: none/lz4/lz4hc
  --target-size N        Target file size in MB (auto-selects quant type)
```

---

## Phase 15: Testing Plan

### 15.1 Unit Tests
```
For each kernel:
  - Test with known input/output pairs
  - Test with zero input
  - Test with max/min values
  - Test alignment edge cases (unaligned input)
  - Test all size classes (1 element, 63 elements, 64 elements, 65 elements, etc.)
  - Compare output against naive scalar implementation
  - Maximum allowed error: 1e-3 for 4-bit, 1e-2 for 2-bit

For quantization:
  - Round-trip test: quantize → dequantize → compare with original
  - Test with uniform distribution
  - Test with normal distribution
  - Test with outliers
  - Test block boundaries

For file format:
  - Write → read round-trip
  - Corrupted file detection (bad magic, bad CRC)
  - Truncated file handling

For KV cache:
  - Store → retrieve correctness
  - Window rotation
  - Full cache behavior

For memory management:
  - Arena doesn't exceed budget
  - Alignment is correct
  - Double buffer state machine transitions
```

### 15.2 Integration Tests
```
  - Load GPT-2 small (124M), generate text, verify coherence
  - Compare output with HuggingFace reference (first 10 tokens should match
    with greedy sampling)
  - Measure perplexity on WikiText-2, compare with published results
  - Verify RAM stays under budget during entire generation
  - Test with sequence length 1, 128, 512, max
  - Test prompt processing + generation handoff
```

### 15.3 Benchmarks
```
  - Tokens per second (generation mode, single token at a time)
  - Prompt processing speed (tokens per second, batched)
  - Time breakdown: I/O, decompress, compute, attention, other
  - Memory high-water mark
  - Compare with llama.cpp at same quantization level
```

---

## Phase 16: Build System

### 16.1 Build Configuration
```
Single-file build option:
  cc -O3 -march=native -o qsf-run qsf.c -lpthread -llz4 -lm

Production build (Makefile or CMake):
  Separate compilation units:
    qsf_format.c     — file format read/write
    qsf_quant.c      — quantization/dequantization
    qsf_kernel_avx2.c  — AVX2 kernels (compiled with -mavx2 -mfma)
    qsf_kernel_avx512.c — AVX-512 kernels (compiled with -mavx512f)
    qsf_kernel_neon.c  — NEON kernels
    qsf_kernel_scalar.c — Scalar fallback
    qsf_engine.c      — main engine logic
    qsf_memory.c      — arena allocator, memory management
    qsf_threading.c   — thread pool
    qsf_tokenizer.c   — BPE tokenizer
    qsf_platform.c    — platform abstraction
    qsf_kv_cache.c    — KV cache management
    qsf_sample.c      — sampling strategies
    qsf_main.c        — CLI entry point

  Key compiler flags:
    -O3              — maximum optimization
    -march=native    — use all available CPU features
    -flto            — link-time optimization (critical for inlining across files)
    -ffast-math      — allow floating point reordering (careful with NaN handling)
    -DNDEBUG         — disable asserts in release
    -fomit-frame-pointer — free up a register
    -fno-exceptions  — not applicable (C, not C++)

  Dependencies:
    - lz4 (can be bundled, single .c file)
    - pthreads (system)
    - math library (system)
    - NO OTHER DEPENDENCIES

  Target: single static binary, no shared library dependencies
```

---

## Phase 17: File Sizes and Performance Targets

### 17.1 Model Size Estimates
```
Model          | Original   | QSF 2-bit+LZ4 | QSF 4-bit+LZ4 | RAM at runtime
---------------|------------|----------------|----------------|---------------
GPT-2 Small    | 500 MB     | ~40 MB         | ~75 MB         | ~45 MB
GPT-2 Medium   | 1.4 GB     | ~100 MB        | ~190 MB        | ~60 MB
GPT-2 Large    | 3.0 GB     | ~220 MB        | ~400 MB        | ~90 MB
LLaMA-7B       | 14 GB      | ~2.0 GB        | ~3.8 GB        | ~180 MB
LLaMA-13B      | 26 GB      | ~3.8 GB        | ~7.0 GB        | ~200 MB (tight)
Mistral-7B     | 14 GB      | ~2.0 GB        | ~3.8 GB        | ~180 MB
Phi-2 (2.7B)   | 5.4 GB     | ~800 MB        | ~1.5 GB        | ~120 MB
```

### 17.2 Performance Targets
```
Token generation (single token, all layers streamed):

Platform              | GPT-2 Small | GPT-2 Med | LLaMA-7B  
----------------------|-------------|-----------|----------
i7-12700H AVX2        | 100+ tok/s  | 50 tok/s  | 5 tok/s   
i9-13900K AVX-512     | 200+ tok/s  | 80 tok/s  | 8 tok/s   
M2 MacBook NEON       | 120+ tok/s  | 60 tok/s  | 6 tok/s   
Raspberry Pi 5        | 15 tok/s    | 5 tok/s   | <1 tok/s  

Bottleneck analysis:
  Small models: compute-bound (matmul dominates)
  Large models: I/O-bound if on HDD, compute-bound if on SSD

  For 7B model, each layer ≈ 60MB quantized data at 2-bit
  32 layers = ~2GB total reads per token
  SSD at 3 GB/s → 0.7 seconds just for reads
  Compute for 32 layers ≈ 0.3 seconds
  Total: ~1 second per token (1 tok/s) without prefetch overlap
  With double buffering: ~0.7 seconds (I/O overlapped) → ~1.5 tok/s
  With NVMe at 7 GB/s: ~0.3 seconds I/O → ~3 tok/s

  Real bottleneck for 7B: sequential layer execution cannot be parallelized
  Each layer depends on previous layer's output
```

---

## Phase 18: Future Extensions (Not MVP)

```
18.1 Speculative Decoding
  - Run small draft model to predict N tokens
  - Verify with large model (batch verify is faster than one-by-one)
  - Accept correct predictions, reject incorrect ones
  - Net speedup: 2-3x for well-matched draft/target pairs

18.2 Continuous Batching
  - Handle multiple concurrent requests
  - Different requests at different layers → share I/O cost
  - Server mode

18.3 Model Sharding
  - Split model across multiple machines
  - Each machine handles subset of layers
  - Pipeline parallelism over network

18.4 Fine-tuning Support
  - LoRA adapter loading (tiny additional weights)
  - Store LoRA as separate small file
  - Apply LoRA during dequant: W_effective = W_base + scale * A * B

18.5 WASM Compilation
  - Compile everything to WebAssembly
  - Run in browser with WASM SIMD
  - Model file served via HTTP range requests
  - True serverless inference

18.6 Disk-as-Memory Fallback
  - If model doesn't fit in RAM budget at all
  - Use OS page cache as implicit cache
  - mmap entire file, let OS manage what's resident
  - Slower but works for any model size
```

---

## Implementation Order (Priority)

```
Week 1-2:
  ☐ File format specification (Phase 1)
  ☐ Arena allocator (Phase 2.1)
  ☐ Scalar kernels (Phase 4, scalar only)
  ☐ Basic quantization (Phase 3, 4-bit only)
  ☐ Conversion tool for GPT-2 small

Week 3-4:
  ☐ Single-threaded inference engine (Phase 5)
  ☐ KV cache (Phase 7, uncompressed first)
  ☐ BPE tokenizer (Phase 12)
  ☐ CLI tool (Phase 14)
  ☐ Verify: GPT-2 small generates coherent text

Week 5-6:
  ☐ AVX2 kernels (Phase 4)
  ☐ NEON kernels (Phase 4)
  ☐ 2-bit and ternary quantization (Phase 3)
  ☐ LZ4 compression (Phase 6.3)
  ☐ Double-buffered layer streaming (Phase 6)

Week 7-8:
  ☐ KV cache quantization (Phase 7)
  ☐ Thread pool (Phase 11)
  ☐ LLaMA support (RoPE, SiLU, RMSNorm, GQA)
  ☐ AVX-512 kernels (Phase 4)

Week 9-10:
  ☐ Layer skipping (Phase 10)
  ☐ Approximate output projection (Phase 9)
  ☐ Embedding cache (Phase 8)
  ☐ Performance tuning and benchmarking

Week 11-12:
  ☐ Full test suite (Phase 15)
  ☐ Documentation
  ☐ Packaging and release
```

This plan covers every single detail from byte-level file format to OS-level I/O strategy to register-level kernel optimization. Each phase is self-contained and testable independently.