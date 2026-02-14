# QStream

**QStream** is a high-performance, CPU-optimized inference engine for Large Language Models (LLMs), written entirely in C11. Designed to run quantized models efficiently on laptops, desktops, and edge devices with minimal dependencies.

## Key Features

*   **Pure C Implementation**: No C++, Python, or heavy frameworks required at inference time. Depends only on the C standard library.
*   **High Performance**: Hand-optimized SIMD kernels for **AVX2** (x86_64) and **NEON** (ARM64/Apple Silicon).
*   **Extreme Compression**: 2-bit, 3-bit, 4-bit quantization with **outlier-aware** mode — extracts critical weights as FP16 for better accuracy at the same size.
*   **MoE Support**: Full Mixture-of-Experts support (Mixtral, DeepSeek, GPT-OSS) with MXFP4 dequantization.
*   **Custom QSF Format**: **QSF** (Quantized Stream Format) model container with optional LZ4 compression.
*   **Cross-Platform**: Builds natively on Windows (MSVC), Linux (GCC/Clang), and macOS (Apple Silicon + Intel).
*   **Edge-Ready**: Stream layers from disk — only one transformer layer in RAM at a time.
*   **Zero-Copy Loading**: Memory-mapped (mmap) file loading for instant startup.

## Compression vs GGUF

| Format | 7B Model Size | Quality | Memory |
|:---|:---|:---|:---|
| GGUF Q4_K_M | ~4.1 GB | Good | ~5.5 GB |
| QSF 4-bit | ~3.8 GB | Good | ~4.5 GB |
| QSF 2-bit+outlier | ~2.1 GB | Good | ~2.8 GB |
| QSF 2-bit | ~1.9 GB | Fair | ~2.5 GB |

The outlier-aware mode extracts the top 0.5% of weights by magnitude and stores them at FP16 precision, while quantizing the remaining 99.5% at 2-bit with a tighter range. This gives quality comparable to 4-bit at nearly half the size.

---

## Quick Start

### 1. Build QStream

#### Prerequisites

*   **CMake** 3.16+
*   **C11 Compiler**: GCC 7+, Clang 6+, or MSVC 2019+
*   **LZ4** (optional, for compressed models): `sudo apt install liblz4-dev`

#### Linux / macOS

```bash
git clone https://github.com/user/qstream.git
cd qstream
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

#### macOS (Apple Silicon)

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)
```

NEON kernels are automatically enabled on ARM64.

#### Windows (Visual Studio)

```powershell
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

#### Windows (MinGW-w64)

```bash
mkdir build && cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
mingw32-make -j
```

The build produces:
- `qstream` (or `qstream.exe`) — CLI inference tool
- `libqsf_lib.a` — static library for embedding

### 2. Convert a HuggingFace Model to QSF

#### Prerequisites

```bash
pip install numpy safetensors torch   # torch is optional but recommended
pip install lz4 tqdm                  # optional: compression + progress bar
```

#### Basic Conversion

```bash
# 4-bit quantization (recommended for most models)
python tools/convert_hf.py ./path/to/hf-model/ -o model.qsf --quant 4bit

# 2-bit quantization (smallest size, good for edge)
python tools/convert_hf.py ./path/to/hf-model/ -o model.qsf --quant 2bit

# 2-bit with outlier extraction (best quality/size ratio)
python tools/convert_hf.py ./path/to/hf-model/ -o model.qsf --quant 2bit_outlier --outlier-frac 0.005

# With LZ4 compression (smaller file, slightly slower load)
python tools/convert_hf.py ./path/to/hf-model/ -o model.qsf --quant 4bit --compress
```

#### GPU-Accelerated Conversion

If you have a CUDA GPU, conversion is automatically accelerated:

```bash
# Auto-detect GPU
python tools/convert_hf.py ./model/ -o model.qsf --quant 4bit

# Force specific GPU
python tools/convert_hf.py ./model/ -o model.qsf --quant 4bit --device cuda:0

# Force CPU
python tools/convert_hf.py ./model/ -o model.qsf --quant 4bit --device cpu
```

#### Converting GPT-OSS / MoE Models

GPT-OSS and other MoE models (Mixtral, DeepSeek) with MXFP4 quantized weights are automatically detected and handled:

```bash
# GPT-OSS 20B (32 experts, MXFP4 format)
python tools/convert_hf.py ./gpt-oss-20b/ -o gpt-oss-20b-2bit.qsf --quant 2bit --device cuda

# Mixtral 8x7B
python tools/convert_hf.py ./mixtral-8x7b/ -o mixtral-4bit.qsf --quant 4bit
```

**Estimated conversion times (GPT-OSS 20B):**

| Device | Quant | Time |
|:---|:---|:---|
| CUDA GPU | 4-bit | ~5-6 min |
| CUDA GPU | 2-bit | ~6-8 min |
| CPU only | 4-bit | ~15-20 min |
| CPU only | 2-bit | ~20-25 min |

#### Conversion Options

| Option | Description | Default |
|:---|:---|:---|
| `--quant TYPE` | `2bit`, `3bit`, `4bit`, `4bit_sym`, `2bit_outlier`, `4bit_outlier`, `fp16` | `4bit` |
| `--block-size N` | Quantization block size (32, 64, 128, 256) | `64` |
| `--compress` | LZ4-compress layer data | Off |
| `--outlier-frac F` | Fraction of outlier weights to extract as FP16 | `0.0` |
| `--device DEV` | `cpu`, `cuda`, `cuda:0` | Auto-detect |
| `-v` | Verbose output | Off |

### 3. Run Inference

```bash
# Interactive chat
./build/qstream model.qsf

# Single prompt
./build/qstream model.qsf -p "Once upon a time" -n 200

# Greedy decoding (deterministic)
./build/qstream model.qsf -p "def fibonacci(n):" -t 0

# Adjust sampling
./build/qstream model.qsf -p "Explain gravity" -t 0.8 -k 50 --top-p 0.95 -n 500
```

#### Inference Options

| Option | Description | Default |
|:---|:---|:---|
| `-p, --prompt "TEXT"` | Input prompt (omit for interactive mode) | Interactive |
| `-n, --max-tokens N` | Max tokens to generate | 256 |
| `-t, --temperature F` | Sampling temperature (0 = greedy) | 0.7 |
| `-k, --top-k N` | Top-K sampling | 40 |
| `--top-p F` | Nucleus sampling threshold | 0.9 |
| `--seed N` | RNG seed for reproducibility | 42 |
| `--no-mmap` | Disable memory-mapping (load full model to RAM) | mmap on |
| `--threads N` | Number of CPU threads (0 = auto) | Auto |
| `--budget N` | RAM budget in bytes (0 = auto) | Auto |
| `-v, --verbose` | Verbose output (use twice for debug) | Off |

---

## Supported Models

| Architecture | Models | Status |
|:---|:---|:---|
| LLaMA | LLaMA 1/2/3, CodeLLaMA, TinyLLaMA | Fully supported |
| Mistral | Mistral 7B, Mistral Nemo | Fully supported |
| Mixtral | Mixtral 8x7B, 8x22B | MoE supported |
| GPT-OSS | GPT-OSS 20B (MXFP4) | MoE + MXFP4 supported |
| Phi | Phi-1/2/3 | Fully supported |
| GPT-2 | GPT-2 (all sizes) | Fully supported |
| GPT-J | GPT-J-6B | Fully supported |
| Qwen | Qwen2 | Fully supported |
| Gemma | Gemma 2B/7B | Fully supported |
| DeepSeek | DeepSeek V2 | MoE supported |

## Edge Device Deployment

QStream is designed for resource-constrained environments:

**Minimum requirements for a 7B model:**
- 2-bit: ~2 GB RAM, any x86_64 or ARM64 CPU
- 4-bit: ~4 GB RAM

**Recommended for best speed:**
- CPU with AVX2 (Intel Haswell+ / AMD Zen+) or NEON (any ARM64)
- SSD storage for fast layer streaming

**Memory optimization tips:**
1. Use `--no-mmap` if your OS has limited virtual memory
2. Use `2bit` or `2bit_outlier` quantization for smallest footprint
3. The engine streams one layer at a time — total model size does not need to fit in RAM
4. Use `--budget` to cap memory usage explicitly

---

## Project Structure

```
qstream/
  include/qsf/    Public API headers
    types.h        On-disk format definitions
    engine.h       Inference engine API
    kernels.h      SIMD kernel dispatch
    quant.h        Quantization/dequantization
    format.h       QSF file reader
    ...
  src/             Core implementation
    engine.c       Transformer inference pipeline
    kernels_avx2.c AVX2 SIMD kernels
    kernels_neon.c ARM NEON kernels
    kernels_scalar.c Portable scalar fallback
    quant.c        Block quantization + outlier-aware
    format.c       QSF file parser
    main.c         CLI entry point
    ...
  tools/
    convert_hf.py  HuggingFace -> QSF converter
  tests/
    test_core.c    Unit tests
  CMakeLists.txt   Build configuration
```

## QSF File Format

The QSF (Quantized Stream Format) is a custom binary format optimized for streaming inference:

```
[Header: 256 bytes]
[Layer Index: 48 bytes × num_layers]
[Embedding Section]
[Tokenizer Section]
[Layer 0 Data] [Layer 1 Data] ... [Layer N Data]
[Final Section: norm weights + output head]
```

Each layer is self-contained and can be loaded/decompressed independently, enabling streaming inference with minimal RAM.


