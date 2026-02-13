# QStream

**QStream** is a high-performance, CPU-optimized inference engine for Large Language Models (LLMs), written entirely in C11. It is designed to run modern quantized models efficiently on consumer hardware with minimal dependencies.

## Key Features

*   **Pure C Implementation**: No C++, Python, or heavy frameworks (PyTorch/TensorFlow) required. Depends only on the C standard library.
*   **High Performance**: Hand-optimized SIMD kernels for **AVX2** (x86_64) and **NEON** (ARM64/Apple Silicon).
*   **Memory Efficient**: Supports 2-bit, 3-bit, and 4-bit block-wise quantization to dramatically reduce RAM usage.
*   **Custom QSF Format**: Uses the **QSF** (Quantized Stream Format) model container, supporting LZ4 compression for fast loading and smaller disk footprint.
*   **Cross-Platform**: Builds and runs natively on Windows (MSVC), Linux (GCC/Clang), and macOS.
*   **Zero-Copy Loading**: Supports memory-mapped (mmap) file loading for instant startup times.

## Build Instructions

### Prerequisites

*   **CMake** 3.16 or later
*   **C Compiler** compatible with C11 (GCC, Clang, MSVC)
*   **LZ4** (Optional, but recommended for compressed models)

### Linux & macOS

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

### Windows (PowerShell)

```powershell
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

_Note: If `LZ4` libraries are not found, the build will proceed without compression support._

## Usage

The primary interface is the `qstream` command-line tool.

```bash
./qstream <model.qsf> [options]
```

### Common Options

| Option | Description | Default |
| :--- | :--- | :--- |
| `-p, --prompt "TEXT"` | Input prompt. If omitted, enters interactive mode. | (Interactive) |
| `-n, --max-tokens N` | Maximum number of tokens to generate. | 256 |
| `-t, --temperature F` | Sampling temperature (0.0 = greedy). | 0.7 |
| `-k, --top-k N` | Top-K sampling. | 40 |
| `--top-p F` | Top-P (Nucleus) sampling. | 0.9 |
| `--seed N` | Random number generator seed. | 42 |
| `--no-mmap` | Disable memory mapping (load entire model to RAM). | Enabled |
| `-v, --verbose` | Enable verbose output (use twice for debug). | Off |

### Examples

**Interactive Mode:**
```bash
./qstream models/llama2-7b.qsf
```

**Single Prompt Generation:**
```bash
./qstream models/mistral-7b.qsf -p "Explain quantum computing in simple terms" -t 0.8 -n 512
```

**Greedy Decoding (Deterministic):**
```bash
./qstream models/tinyllama.qsf -p "def fibonacci(n):" -t 0
```

## Project Structure

*   `src/` - Core source code (engine, kernels, quantization).
*   `include/qsf/` - Public API headers.
*   `tools/` - Utility scripts (e.g., model converters).
*   `tests/` - Unit tests.
*   `CMakeLists.txt` - Build configuration.


