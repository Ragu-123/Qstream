#!/usr/bin/env python3
"""
QStream - convert_hf.py
Convert HuggingFace transformer models to QSF format.

Supports: GPT-2, LLaMA, LLaMA-2, Mistral, Phi, GPT-J
Input:  HuggingFace model directory (safetensors or pytorch_model.bin)
Output: Single .qsf binary file

Usage:
    python convert_hf.py <model_dir> -o model.qsf [--quant 4bit]
"""

import os
import sys
import json
import struct
import hashlib
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    import lz4.block
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger('convert_hf')

# ── QSF Constants ────────────────────────────────────────────────────
QSF_MAGIC       = 0x51534631  # "QSF1"
QSF_VERSION     = 1
QSF_HEADER_SIZE = 256
QSF_ALIGNMENT   = 64
QSF_NO_TOKEN    = 0xFFFFFFFF

# Architecture IDs
ARCH_MAP = {
    'gpt2': 0, 'llama': 1, 'mistral': 2, 'phi': 3,
    'gptj': 4, 'gpt_neox': 4, 'qwen2': 5, 'gemma': 6,
    'stablelm': 7,
}

# Quantization type IDs
QUANT_2BIT_ASYM  = 0
QUANT_2BIT_SYM   = 1
QUANT_3BIT_ASYM  = 2
QUANT_4BIT_ASYM  = 3
QUANT_4BIT_SYM   = 4
QUANT_FP16       = 7

# Activation IDs
ACT_MAP = {
    'gelu': 0, 'gelu_new': 1, 'gelu_fast': 1,
    'silu': 2, 'swish': 2,
    'relu': 3, 'relu2': 4, 'geglu': 5,
}

# Norm type IDs
NORM_MAP = {
    'layernorm_pre': 0, 'layernorm_post': 1,
    'rmsnorm_pre': 2, 'rmsnorm_post': 3,
}

# FFN type IDs
FFN_STANDARD = 0
FFN_GATED    = 1
FFN_PARALLEL = 2

# Tensor type IDs
TENSOR_ATTN_Q      = 0
TENSOR_ATTN_K      = 1
TENSOR_ATTN_V      = 2
TENSOR_ATTN_O      = 3
TENSOR_FFN_GATE    = 4
TENSOR_FFN_UP      = 5
TENSOR_FFN_DOWN    = 6
TENSOR_ATTN_NORM_W = 7
TENSOR_ATTN_NORM_B = 8
TENSOR_FFN_NORM_W  = 9
TENSOR_FFN_NORM_B  = 10
TENSOR_ATTN_Q_BIAS = 11
TENSOR_ATTN_K_BIAS = 12
TENSOR_ATTN_V_BIAS = 13
TENSOR_ATTN_O_BIAS = 14
TENSOR_FFN_GATE_BIAS = 15
TENSOR_FFN_UP_BIAS   = 16
TENSOR_FFN_DOWN_BIAS = 17
TENSOR_FUSED_QKV     = 18
TENSOR_FUSED_QKV_BIAS = 19
TENSOR_POS_EMBED     = 20


###############################################################################
# CRC-32 (matches format.c)
###############################################################################
def crc32(data: bytes) -> int:
    """Standard CRC-32 computation matching the C implementation."""
    import binascii
    return binascii.crc32(data) & 0xFFFFFFFF


###############################################################################
# Quantization
###############################################################################
class Quantizer:
    """Block quantizer for 2/3/4-bit quantization."""

    def __init__(self, quant_type: int, block_size: int = 64):
        self.quant_type = quant_type
        self.block_size = block_size

    def quantize_tensor(self, tensor: np.ndarray) -> Tuple[bytes, int, int]:
        """Quantize a 2D tensor. Returns (packed_bytes, rows, cols)."""
        if tensor.ndim == 1:
            tensor = tensor.reshape(1, -1)
        rows, cols = tensor.shape
        flat = tensor.flatten().astype(np.float32)
        bs = self.block_size
        num_blocks = (len(flat) + bs - 1) // bs
        chunks = []

        for i in range(num_blocks):
            start = i * bs
            end = min(start + bs, len(flat))
            block = flat[start:end]
            packed = self._quantize_block(block)
            chunks.append(packed)

        return b''.join(chunks), rows, cols

    def _quantize_block(self, block: np.ndarray) -> bytes:
        """Quantize a single block."""
        if self.quant_type == QUANT_4BIT_ASYM:
            return self._quant_4bit_asym(block)
        elif self.quant_type == QUANT_4BIT_SYM:
            return self._quant_4bit_sym(block)
        elif self.quant_type == QUANT_2BIT_ASYM:
            return self._quant_2bit_asym(block)
        elif self.quant_type == QUANT_3BIT_ASYM:
            return self._quant_3bit_asym(block)
        elif self.quant_type == QUANT_FP16:
            return self._quant_fp16(block)
        else:
            raise ValueError(f"Unknown quant type: {self.quant_type}")

    def _float_to_fp16(self, val: float) -> int:
        """Convert float32 to IEEE fp16 as uint16."""
        return int(np.float16(val).view(np.uint16))

    def _quant_4bit_asym(self, block: np.ndarray) -> bytes:
        """4-bit asymmetric: scale(fp16) + min(fp16) + packed nibbles."""
        n = len(block)
        bmin = float(block.min())
        bmax = float(block.max())
        scale = (bmax - bmin) / 15.0 if bmax != bmin else 1.0
        inv_scale = 1.0 / scale if scale != 0 else 0.0

        # Header: scale(fp16) + min(fp16) = 4 bytes
        header = struct.pack('<HH', self._float_to_fp16(scale),
                             self._float_to_fp16(bmin))

        # Quantize to 0-15 and pack two per byte
        vals = np.clip(np.round((block - bmin) * inv_scale), 0, 15).astype(np.uint8)
        packed_bytes = bytearray()
        for j in range(0, n, 2):
            lo = vals[j]
            hi = vals[j + 1] if j + 1 < n else 0
            packed_bytes.append(lo | (hi << 4))

        return header + bytes(packed_bytes)

    def _quant_4bit_sym(self, block: np.ndarray) -> bytes:
        """4-bit symmetric: scale(fp16) + packed nibbles (values are -7..+7)."""
        n = len(block)
        absmax = float(np.abs(block).max())
        scale = absmax / 7.0 if absmax > 0 else 1.0
        inv_scale = 1.0 / scale if scale != 0 else 0.0

        header = struct.pack('<H', self._float_to_fp16(scale))

        # Quantize to -7..+7, store as unsigned 0..15 (offset by 8)
        vals = np.clip(np.round(block * inv_scale), -7, 7).astype(np.int8) + 8
        vals = vals.astype(np.uint8)
        packed_bytes = bytearray()
        for j in range(0, n, 2):
            lo = vals[j]
            hi = vals[j + 1] if j + 1 < n else 8
            packed_bytes.append(lo | (hi << 4))

        return header + bytes(packed_bytes)

    def _quant_2bit_asym(self, block: np.ndarray) -> bytes:
        """2-bit asymmetric: scale(fp16) + min(fp16) + packed 2-bit values."""
        n = len(block)
        bmin = float(block.min())
        bmax = float(block.max())
        scale = (bmax - bmin) / 3.0 if bmax != bmin else 1.0
        inv_scale = 1.0 / scale if scale != 0 else 0.0

        header = struct.pack('<HH', self._float_to_fp16(scale),
                             self._float_to_fp16(bmin))

        vals = np.clip(np.round((block - bmin) * inv_scale), 0, 3).astype(np.uint8)
        packed_bytes = bytearray()
        for j in range(0, n, 4):
            byte_val = 0
            for k in range(4):
                if j + k < n:
                    byte_val |= (vals[j + k] & 0x03) << (k * 2)
            packed_bytes.append(byte_val)

        return header + bytes(packed_bytes)

    def _quant_3bit_asym(self, block: np.ndarray) -> bytes:
        """3-bit asymmetric: scale(fp16) + min(fp16) + packed 3-bit values."""
        n = len(block)
        bmin = float(block.min())
        bmax = float(block.max())
        scale = (bmax - bmin) / 7.0 if bmax != bmin else 1.0
        inv_scale = 1.0 / scale if scale != 0 else 0.0

        header = struct.pack('<HH', self._float_to_fp16(scale),
                             self._float_to_fp16(bmin))

        vals = np.clip(np.round((block - bmin) * inv_scale), 0, 7).astype(np.uint8)
        # Pack 3-bit values into bytes (8 values → 3 bytes)
        packed_bytes = bytearray()
        bits = 0
        bit_count = 0
        for v in vals:
            bits |= (v & 0x07) << bit_count
            bit_count += 3
            while bit_count >= 8:
                packed_bytes.append(bits & 0xFF)
                bits >>= 8
                bit_count -= 8
        if bit_count > 0:
            packed_bytes.append(bits & 0xFF)

        return header + bytes(packed_bytes)

    def _quant_fp16(self, block: np.ndarray) -> bytes:
        """Store as raw FP16."""
        return block.astype(np.float16).tobytes()

    def block_data_size(self, num_elements: int) -> int:
        """Estimate total quantized size for num_elements."""
        bs = self.block_size
        num_blocks = (num_elements + bs - 1) // bs
        if self.quant_type in (QUANT_4BIT_ASYM,):
            return num_blocks * (4 + bs // 2)
        elif self.quant_type == QUANT_4BIT_SYM:
            return num_blocks * (2 + bs // 2)
        elif self.quant_type == QUANT_2BIT_ASYM:
            return num_blocks * (4 + bs // 4)
        elif self.quant_type == QUANT_3BIT_ASYM:
            bits = bs * 3
            data_bytes = (bits + 7) // 8
            return num_blocks * (4 + data_bytes)
        elif self.quant_type == QUANT_FP16:
            return num_elements * 2
        return num_elements * 4


###############################################################################
# Architecture-specific tensor name mapping
###############################################################################
def get_tensor_mapping(arch: str, num_layers: int) -> Dict[str, Tuple[str, int]]:
    """
    Build a mapping: HF_tensor_name → (qsf_role, layer_index).
    layer_index = -1 for global tensors (embedding, final norm, output head).
    Returns mapping and set of expected tensor names.
    """
    mapping = {}

    if arch == 'gpt2':
        mapping['transformer.wte.weight'] = ('embedding', -1)
        mapping['transformer.wpe.weight'] = ('pos_embed', -1)
        mapping['transformer.ln_f.weight'] = ('final_norm_w', -1)
        mapping['transformer.ln_f.bias'] = ('final_norm_b', -1)
        for L in range(num_layers):
            p = f'transformer.h.{L}'
            mapping[f'{p}.ln_1.weight'] = ('attn_norm_w', L)
            mapping[f'{p}.ln_1.bias'] = ('attn_norm_b', L)
            mapping[f'{p}.attn.c_attn.weight'] = ('fused_qkv_w', L)
            mapping[f'{p}.attn.c_attn.bias'] = ('fused_qkv_b', L)
            mapping[f'{p}.attn.c_proj.weight'] = ('attn_o_w', L)
            mapping[f'{p}.attn.c_proj.bias'] = ('attn_o_b', L)
            mapping[f'{p}.ln_2.weight'] = ('ffn_norm_w', L)
            mapping[f'{p}.ln_2.bias'] = ('ffn_norm_b', L)
            mapping[f'{p}.mlp.c_fc.weight'] = ('ffn_up_w', L)
            mapping[f'{p}.mlp.c_fc.bias'] = ('ffn_up_b', L)
            mapping[f'{p}.mlp.c_proj.weight'] = ('ffn_down_w', L)
            mapping[f'{p}.mlp.c_proj.bias'] = ('ffn_down_b', L)

    elif arch in ('llama', 'mistral', 'qwen2'):
        mapping['model.embed_tokens.weight'] = ('embedding', -1)
        mapping['model.norm.weight'] = ('final_norm_w', -1)
        mapping['lm_head.weight'] = ('output_head', -1)
        for L in range(num_layers):
            p = f'model.layers.{L}'
            mapping[f'{p}.self_attn.q_proj.weight'] = ('attn_q_w', L)
            mapping[f'{p}.self_attn.k_proj.weight'] = ('attn_k_w', L)
            mapping[f'{p}.self_attn.v_proj.weight'] = ('attn_v_w', L)
            mapping[f'{p}.self_attn.o_proj.weight'] = ('attn_o_w', L)
            mapping[f'{p}.mlp.gate_proj.weight'] = ('ffn_gate_w', L)
            mapping[f'{p}.mlp.up_proj.weight'] = ('ffn_up_w', L)
            mapping[f'{p}.mlp.down_proj.weight'] = ('ffn_down_w', L)
            mapping[f'{p}.input_layernorm.weight'] = ('attn_norm_w', L)
            mapping[f'{p}.post_attention_layernorm.weight'] = ('ffn_norm_w', L)
            # Optional biases
            for bias_name in ['q_proj.bias', 'k_proj.bias', 'v_proj.bias', 'o_proj.bias']:
                key = f'{p}.self_attn.{bias_name}'
                role = 'attn_' + bias_name.split('.')[0].replace('_proj', '') + '_b'
                mapping[key] = (role, L)

    elif arch == 'phi':
        mapping['model.embed_tokens.weight'] = ('embedding', -1)
        mapping['model.final_layernorm.weight'] = ('final_norm_w', -1)
        mapping['model.final_layernorm.bias'] = ('final_norm_b', -1)
        mapping['lm_head.weight'] = ('output_head', -1)
        mapping['lm_head.bias'] = ('output_head_bias', -1)
        for L in range(num_layers):
            p = f'model.layers.{L}'
            mapping[f'{p}.self_attn.q_proj.weight'] = ('attn_q_w', L)
            mapping[f'{p}.self_attn.k_proj.weight'] = ('attn_k_w', L)
            mapping[f'{p}.self_attn.v_proj.weight'] = ('attn_v_w', L)
            mapping[f'{p}.self_attn.dense.weight'] = ('attn_o_w', L)
            mapping[f'{p}.self_attn.q_proj.bias'] = ('attn_q_b', L)
            mapping[f'{p}.self_attn.k_proj.bias'] = ('attn_k_b', L)
            mapping[f'{p}.self_attn.v_proj.bias'] = ('attn_v_b', L)
            mapping[f'{p}.self_attn.dense.bias'] = ('attn_o_b', L)
            mapping[f'{p}.mlp.fc1.weight'] = ('ffn_up_w', L)
            mapping[f'{p}.mlp.fc1.bias'] = ('ffn_up_b', L)
            mapping[f'{p}.mlp.fc2.weight'] = ('ffn_down_w', L)
            mapping[f'{p}.mlp.fc2.bias'] = ('ffn_down_b', L)
            mapping[f'{p}.input_layernorm.weight'] = ('attn_norm_w', L)
            mapping[f'{p}.input_layernorm.bias'] = ('attn_norm_b', L)

    elif arch in ('gptj', 'gpt_neox'):
        mapping['transformer.wte.weight'] = ('embedding', -1)
        mapping['transformer.ln_f.weight'] = ('final_norm_w', -1)
        mapping['transformer.ln_f.bias'] = ('final_norm_b', -1)
        mapping['lm_head.weight'] = ('output_head', -1)
        mapping['lm_head.bias'] = ('output_head_bias', -1)
        for L in range(num_layers):
            p = f'transformer.h.{L}'
            mapping[f'{p}.ln_1.weight'] = ('attn_norm_w', L)
            mapping[f'{p}.ln_1.bias'] = ('attn_norm_b', L)
            mapping[f'{p}.attn.q_proj.weight'] = ('attn_q_w', L)
            mapping[f'{p}.attn.k_proj.weight'] = ('attn_k_w', L)
            mapping[f'{p}.attn.v_proj.weight'] = ('attn_v_w', L)
            mapping[f'{p}.attn.out_proj.weight'] = ('attn_o_w', L)
            mapping[f'{p}.mlp.fc_in.weight'] = ('ffn_up_w', L)
            mapping[f'{p}.mlp.fc_in.bias'] = ('ffn_up_b', L)
            mapping[f'{p}.mlp.fc_out.weight'] = ('ffn_down_w', L)
            mapping[f'{p}.mlp.fc_out.bias'] = ('ffn_down_b', L)

    return mapping


###############################################################################
# Tensor loading (safetensors or pytorch)
###############################################################################
class TensorLoader:
    """Lazy tensor loader supporting safetensors and pytorch formats."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.tensors = {}      # name → (file, key)
        self._detect_format()

    def _detect_format(self):
        """Detect model format and build tensor index."""
        st_files = sorted(self.model_dir.glob('*.safetensors'))
        pt_files = sorted(self.model_dir.glob('pytorch_model*.bin'))

        if st_files and HAS_SAFETENSORS:
            self.format = 'safetensors'
            self._index_safetensors(st_files)
        elif pt_files and HAS_TORCH:
            self.format = 'pytorch'
            self._index_pytorch(pt_files)
        else:
            raise RuntimeError(
                f"No model files found in {self.model_dir}. "
                f"Need .safetensors (safetensors pkg) or pytorch_model.bin (torch pkg).")

    def _index_safetensors(self, files: List[Path]):
        """Build index from safetensors files."""
        for f in files:
            with safe_open(str(f), framework='numpy') as st:
                for key in st.keys():
                    self.tensors[key] = (str(f), key)
        log.info(f"Indexed {len(self.tensors)} tensors from {len(files)} safetensors file(s)")

    def _index_pytorch(self, files: List[Path]):
        """Build index from pytorch bin files."""
        # Check for shard index
        index_file = self.model_dir / 'pytorch_model.bin.index.json'
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
            for key, shard_file in index['weight_map'].items():
                self.tensors[key] = (str(self.model_dir / shard_file), key)
        else:
            for f in files:
                sd = torch.load(str(f), map_location='cpu', weights_only=True)
                for key in sd.keys():
                    self.tensors[key] = (str(f), key)
                del sd
        log.info(f"Indexed {len(self.tensors)} tensors from pytorch file(s)")

    def get_tensor(self, name: str) -> Optional[np.ndarray]:
        """Load a single tensor by name, return as float32 numpy array."""
        if name not in self.tensors:
            return None
        filepath, key = self.tensors[name]

        if self.format == 'safetensors':
            with safe_open(filepath, framework='numpy') as st:
                arr = st.get_tensor(key)
        else:
            sd = torch.load(filepath, map_location='cpu', weights_only=True)
            arr = sd[key].numpy()
            del sd

        # Convert to float32
        if arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        elif hasattr(np, 'bfloat16') or str(arr.dtype) == 'bfloat16':
            # bfloat16 → float32
            arr = arr.astype(np.float32)

        return arr.astype(np.float32)

    def has_tensor(self, name: str) -> bool:
        return name in self.tensors


###############################################################################
# Tokenizer parsing
###############################################################################
def parse_tokenizer(model_dir: Path) -> dict:
    """Parse HuggingFace tokenizer files into QSF tokenizer data."""
    tok_data = {
        'type': 0,  # BPE
        'vocab': {},
        'merges': [],
        'bos_id': QSF_NO_TOKEN,
        'eos_id': QSF_NO_TOKEN,
        'pad_id': QSF_NO_TOKEN,
        'unk_id': QSF_NO_TOKEN,
    }

    # Try tokenizer.json first
    tok_json = model_dir / 'tokenizer.json'
    if tok_json.exists():
        log.info("Parsing tokenizer.json")
        with open(tok_json, encoding='utf-8') as f:
            tj = json.load(f)

        # Vocab
        if 'model' in tj and 'vocab' in tj['model']:
            tok_data['vocab'] = tj['model']['vocab']

        # Merges
        if 'model' in tj and 'merges' in tj['model']:
            tok_data['merges'] = tj['model']['merges']

        # Added tokens
        if 'added_tokens' in tj:
            for at in tj['added_tokens']:
                tid = at.get('id', len(tok_data['vocab']))
                tok_data['vocab'][at['content']] = tid

    # Parse special tokens
    special_map_file = model_dir / 'special_tokens_map.json'
    tok_config_file = model_dir / 'tokenizer_config.json'

    if tok_config_file.exists():
        with open(tok_config_file, encoding='utf-8') as f:
            tc = json.load(f)
        for key, field in [('bos_token', 'bos_id'), ('eos_token', 'eos_id'),
                           ('pad_token', 'pad_id'), ('unk_token', 'unk_id')]:
            token = tc.get(key)
            if isinstance(token, dict):
                token = token.get('content', token.get('value'))
            if token and token in tok_data['vocab']:
                tok_data[field] = tok_data['vocab'][token]

    elif special_map_file.exists():
        with open(special_map_file, encoding='utf-8') as f:
            sm = json.load(f)
        for key, field in [('bos_token', 'bos_id'), ('eos_token', 'eos_id'),
                           ('pad_token', 'pad_id'), ('unk_token', 'unk_id')]:
            token = sm.get(key)
            if isinstance(token, dict):
                token = token.get('content')
            if token and token in tok_data['vocab']:
                tok_data[field] = tok_data['vocab'][token]

    log.info(f"Tokenizer: {len(tok_data['vocab'])} vocab, "
             f"{len(tok_data['merges'])} merges, "
             f"BOS={tok_data['bos_id']}, EOS={tok_data['eos_id']}")
    return tok_data


def serialize_tokenizer(tok_data: dict) -> bytes:
    """Serialize tokenizer into QSF binary format."""
    # Vocab data: list of (token_string, token_id)
    vocab_items = sorted(tok_data['vocab'].items(), key=lambda x: x[1])
    vocab_buf = bytearray()
    for token_str, token_id in vocab_items:
        token_bytes = token_str.encode('utf-8', errors='replace')
        # Length-prefixed: uint16 len + bytes
        vocab_buf += struct.pack('<H', len(token_bytes))
        vocab_buf += token_bytes

    # Merge data: list of "tokenA tokenB" strings
    merge_buf = bytearray()
    for merge_str in tok_data['merges']:
        parts = merge_str.split(' ', 1)
        if len(parts) == 2:
            a = tok_data['vocab'].get(parts[0], 0)
            b = tok_data['vocab'].get(parts[1], 0)
            # Find merged token
            merged_str = parts[0] + parts[1]
            merged_id = tok_data['vocab'].get(merged_str, 0)
            merge_buf += struct.pack('<III', a, b, merged_id)

    num_merges = len(tok_data['merges'])

    # Tokenizer header (32 bytes)
    header = struct.pack('<IIIIIIII',
        tok_data['type'],           # tokenizer_type
        len(vocab_items),           # vocab_size
        num_merges,                 # num_merges
        0,                          # num_added_tokens
        len(vocab_buf),             # vocab_data_size
        len(merge_buf),             # merge_data_size
        0,                          # added_tokens_data_size
        0,                          # flags
    )

    return header + bytes(vocab_buf) + bytes(merge_buf)


###############################################################################
# QSF File Writer
###############################################################################
class QSFWriter:
    """Writes QSF format files."""

    def __init__(self, output_path: str, config: dict, quant_type: int,
                 block_size: int = 64, compress: bool = False):
        self.output_path = output_path
        self.config = config
        self.quant_type = quant_type
        self.block_size = block_size
        self.compress = compress and HAS_LZ4
        self.quantizer = Quantizer(quant_type, block_size)
        self.norm_quantizer = Quantizer(QUANT_FP16, block_size)

    def write(self, loader: TensorLoader, arch: str, tok_data: dict):
        """Write complete QSF file."""
        cfg = self.config
        num_layers = cfg['num_hidden_layers']

        log.info(f"Writing QSF: {arch}, {num_layers} layers, "
                 f"quant={self.quant_type}, bs={self.block_size}")

        with open(self.output_path, 'wb') as f:
            # Reserve space for header
            f.write(b'\x00' * QSF_HEADER_SIZE)

            # ── 1. Write layer index placeholder ──
            layer_index_offset = self._align_offset(f)
            layer_entries = []
            for _ in range(num_layers):
                f.write(b'\x00' * 48)  # placeholder
            layer_index_size = num_layers * 48

            # ── 2. Write embedding section ──
            embedding_offset = self._align_offset(f)
            self._write_embedding(f, loader, arch, cfg)

            # ── 3. Write tokenizer section ──
            tokenizer_offset = self._align_offset(f)
            tok_bytes = serialize_tokenizer(tok_data)
            f.write(tok_bytes)

            # ── 4. Write each layer ──
            mapping = get_tensor_mapping(arch, num_layers)
            for layer_idx in range(num_layers):
                log.info(f"  Layer {layer_idx}/{num_layers-1}")
                entry = self._write_layer(f, loader, mapping,
                                          layer_idx, arch, cfg)
                layer_entries.append(entry)

            # ── 5. Write final section (norm + output head) ──
            final_offset = self._align_offset(f)
            self._write_final(f, loader, mapping, arch, cfg)

            # ── 6. Record total file size ──
            total_file_size = f.tell()

            # ── 7. Go back and write the real header ──
            f.seek(0)
            header = self._build_header(cfg, arch, tok_data,
                                         layer_index_offset, embedding_offset,
                                         tokenizer_offset, final_offset,
                                         total_file_size)
            f.write(header)

            # ── 8. Write real layer index ──
            f.seek(layer_index_offset)
            for entry in layer_entries:
                f.write(entry)

        log.info(f"Done! Output: {self.output_path} "
                 f"({total_file_size / 1024 / 1024:.1f} MB)")

    def _align_offset(self, f) -> int:
        """Advance file position to next 64-byte aligned offset."""
        pos = f.tell()
        aligned = (pos + QSF_ALIGNMENT - 1) & ~(QSF_ALIGNMENT - 1)
        if aligned > pos:
            f.write(b'\x00' * (aligned - pos))
        return aligned

    def _build_header(self, cfg, arch, tok_data,
                      layer_index_offset, embedding_offset,
                      tokenizer_offset, final_offset,
                      total_file_size) -> bytes:
        """Build the 256-byte QSF header."""
        arch_id = ARCH_MAP.get(arch, 128)
        hidden_dim = cfg['hidden_size']
        num_heads = cfg['num_attention_heads']
        num_kv_heads = cfg.get('num_key_value_heads', num_heads)
        head_dim = cfg.get('head_dim', hidden_dim // num_heads)
        act_str = cfg.get('hidden_act', 'silu')
        activation = ACT_MAP.get(act_str, 0)
        rope_theta = cfg.get('rope_theta', 10000.0)
        norm_eps = cfg.get('rms_norm_eps', cfg.get('layer_norm_epsilon', 1e-5))

        # Determine norm type
        if arch in ('gpt2', 'gptj', 'gpt_neox', 'phi'):
            norm_type = NORM_MAP['layernorm_pre']
        else:
            norm_type = NORM_MAP['rmsnorm_pre']

        # Determine pos encoding
        if arch == 'gpt2':
            pos_enc = 0  # learned
        else:
            pos_enc = 1  # RoPE

        # FFN type
        if arch in ('llama', 'mistral', 'qwen2', 'gemma'):
            ffn_type = FFN_GATED
        elif arch in ('gptj',):
            ffn_type = FFN_PARALLEL
        else:
            ffn_type = FFN_STANDARD

        # Attention type
        if arch == 'gpt2':
            attn_type = 1  # fused QKV
        else:
            attn_type = 0  # separate

        # Bias bitfield
        has_bias = 0
        if arch == 'gpt2':
            has_bias = 0x0F  # all biases
        elif arch == 'phi':
            has_bias = 0x03  # QKV + output bias

        # Weight tying
        tie = cfg.get('tie_word_embeddings', arch == 'gpt2')

        # RoPE scaling
        rope_scaling_type = 0
        rope_scaling_factor = 1.0
        rope_scaling_cfg = cfg.get('rope_scaling')
        if rope_scaling_cfg:
            rs_type = rope_scaling_cfg.get('type', '')
            if rs_type == 'linear':
                rope_scaling_type = 1
                rope_scaling_factor = rope_scaling_cfg.get('factor', 1.0)
            elif rs_type == 'ntk':
                rope_scaling_type = 2
            elif rs_type == 'yarn':
                rope_scaling_type = 3
            elif rs_type == 'dynamic':
                rope_scaling_type = 4

        sliding_window = cfg.get('sliding_window', 0) or 0

        # Compute param count estimate
        total_params = (cfg['vocab_size'] * hidden_dim +
                        cfg['num_hidden_layers'] * (
                            4 * hidden_dim * hidden_dim +
                            2 * hidden_dim * cfg.get('intermediate_size', 4 * hidden_dim)
                        ))
        num_params_millions = total_params // 1_000_000

        # Build header bytes
        buf = bytearray(QSF_HEADER_SIZE)
        struct.pack_into('<I', buf, 0, QSF_MAGIC)
        struct.pack_into('<I', buf, 4, QSF_VERSION)
        struct.pack_into('<I', buf, 8, QSF_HEADER_SIZE)
        struct.pack_into('<I', buf, 12, arch_id)
        struct.pack_into('<I', buf, 16, cfg['num_hidden_layers'])
        struct.pack_into('<I', buf, 20, hidden_dim)
        struct.pack_into('<I', buf, 24, num_heads)
        struct.pack_into('<I', buf, 28, num_kv_heads)
        struct.pack_into('<I', buf, 32, cfg['vocab_size'])
        struct.pack_into('<I', buf, 36, cfg.get('max_position_embeddings', 2048))
        struct.pack_into('<I', buf, 40, cfg.get('intermediate_size', 4 * hidden_dim))
        struct.pack_into('<I', buf, 44, head_dim)
        struct.pack_into('<B', buf, 48, self.quant_type)
        struct.pack_into('<B', buf, 49, activation)
        struct.pack_into('<B', buf, 50, norm_type)
        struct.pack_into('<B', buf, 51, pos_enc)
        struct.pack_into('<f', buf, 52, rope_theta)
        struct.pack_into('<f', buf, 56, norm_eps)
        struct.pack_into('<I', buf, 60, self.block_size)

        # Section offsets
        struct.pack_into('<Q', buf, 64, layer_index_offset)
        struct.pack_into('<Q', buf, 72, embedding_offset)
        struct.pack_into('<Q', buf, 80, final_offset)
        struct.pack_into('<Q', buf, 88, tokenizer_offset)
        struct.pack_into('<Q', buf, 96, 0)   # extended config
        struct.pack_into('<Q', buf, 104, 0)  # calibration
        struct.pack_into('<Q', buf, 112, 0)  # importance scores

        # Special tokens
        struct.pack_into('<I', buf, 120, tok_data['bos_id'])
        struct.pack_into('<I', buf, 124, tok_data['eos_id'])
        struct.pack_into('<I', buf, 128, tok_data['pad_id'])
        struct.pack_into('<I', buf, 132, tok_data['unk_id'])

        # Metadata
        struct.pack_into('<I', buf, 136, num_params_millions)
        struct.pack_into('<B', buf, 140, has_bias)
        struct.pack_into('<B', buf, 141, ffn_type)
        struct.pack_into('<B', buf, 142, attn_type)
        struct.pack_into('<B', buf, 143, 1 if tie else 0)
        struct.pack_into('<I', buf, 144, rope_scaling_type)
        struct.pack_into('<f', buf, 148, rope_scaling_factor)
        struct.pack_into('<I', buf, 152, sliding_window)
        struct.pack_into('<I', buf, 156, 1 if tie else 0)

        # Integrity
        struct.pack_into('<Q', buf, 160, total_file_size)
        # CRC32 of bytes 0-167
        header_crc = crc32(bytes(buf[:168]))
        struct.pack_into('<I', buf, 168, header_crc)
        struct.pack_into('<B', buf, 172, 0x01)  # endian marker = LE
        # bytes 173-255 already zero

        return bytes(buf)

    def _write_embedding(self, f, loader: TensorLoader, arch: str, cfg: dict):
        """Write the embedding section."""
        if arch == 'gpt2':
            emb = loader.get_tensor('transformer.wte.weight')
        else:
            emb = loader.get_tensor('model.embed_tokens.weight')

        if emb is None:
            raise RuntimeError("Embedding weights not found!")

        vocab_size, emb_dim = emb.shape
        log.info(f"  Embedding: {vocab_size} x {emb_dim}")

        quant_data, _, _ = self.quantizer.quantize_tensor(emb)
        del emb

        compressed = quant_data
        comp_type = 0
        if self.compress:
            compressed = lz4.block.compress(quant_data, store_size=False)
            comp_type = 1
            log.info(f"    Compressed: {len(quant_data)} → {len(compressed)} bytes")

        # Embedding header (32 bytes)
        emb_header = struct.pack('<IIIIBBBBIIII',
            self.quant_type,        # quant_type
            len(compressed) if comp_type else 0,  # compressed_size
            vocab_size,             # num_vectors
            emb_dim,                # embedding_dim
            comp_type,              # compression_type
            0, 0, 0,               # reserved
            crc32(compressed),      # crc32
            1,                      # num_chunks
            vocab_size,             # chunk_size
            0,                      # padding
        )
        # Fix to 32 bytes
        emb_header = emb_header[:32]

        f.write(emb_header)
        f.write(compressed)

    def _write_layer(self, f, loader: TensorLoader,
                     mapping: Dict, layer_idx: int,
                     arch: str, cfg: dict) -> bytes:
        """Write one transformer layer, return 48-byte layer index entry."""
        offset = self._align_offset(f)
        start_pos = f.tell()

        # Determine which tensors belong to this layer
        role_to_type = {
            'attn_q_w': TENSOR_ATTN_Q, 'attn_k_w': TENSOR_ATTN_K,
            'attn_v_w': TENSOR_ATTN_V, 'attn_o_w': TENSOR_ATTN_O,
            'ffn_gate_w': TENSOR_FFN_GATE, 'ffn_up_w': TENSOR_FFN_UP,
            'ffn_down_w': TENSOR_FFN_DOWN,
            'attn_norm_w': TENSOR_ATTN_NORM_W, 'attn_norm_b': TENSOR_ATTN_NORM_B,
            'ffn_norm_w': TENSOR_FFN_NORM_W, 'ffn_norm_b': TENSOR_FFN_NORM_B,
            'attn_q_b': TENSOR_ATTN_Q_BIAS, 'attn_k_b': TENSOR_ATTN_K_BIAS,
            'attn_v_b': TENSOR_ATTN_V_BIAS, 'attn_o_b': TENSOR_ATTN_O_BIAS,
            'ffn_gate_b': TENSOR_FFN_GATE_BIAS,
            'ffn_up_b': TENSOR_FFN_UP_BIAS, 'ffn_down_b': TENSOR_FFN_DOWN_BIAS,
            'fused_qkv_w': TENSOR_FUSED_QKV, 'fused_qkv_b': TENSOR_FUSED_QKV_BIAS,
        }

        num_tensors = 0
        layer_buf = bytearray()

        for hf_name, (role, lidx) in mapping.items():
            if lidx != layer_idx:
                continue

            tensor_type_id = role_to_type.get(role)
            if tensor_type_id is None:
                continue

            tensor = loader.get_tensor(hf_name)
            if tensor is None:
                continue

            # GPT-2 conv1d weights are transposed
            if arch == 'gpt2' and role in ('fused_qkv_w', 'attn_o_w',
                                            'ffn_up_w', 'ffn_down_w'):
                tensor = tensor.T.copy()

            # Choose quantizer: norms stay in FP16, weights get quantized
            if role in ('attn_norm_w', 'attn_norm_b', 'ffn_norm_w', 'ffn_norm_b'):
                q = self.norm_quantizer
                qt = QUANT_FP16
            else:
                q = self.quantizer
                qt = self.quant_type

            quant_data, rows, cols = q.quantize_tensor(tensor)
            del tensor

            # Tensor header (24 bytes)
            tensor_hdr = struct.pack('<HBBIIIII',
                tensor_type_id,     # tensor_type
                qt,                 # quant_type
                0,                  # data_layout (row-major)
                rows,               # rows
                cols,               # cols
                len(quant_data),    # data_size
                0,                  # num_outliers
                0,                  # reserved
            )

            layer_buf += tensor_hdr
            layer_buf += quant_data
            num_tensors += 1

        # Optionally compress
        layer_bytes = bytes(layer_buf)
        decompressed_size = len(layer_bytes)
        comp_type = 0
        if self.compress and decompressed_size > 1024:
            compressed = lz4.block.compress(layer_bytes, store_size=False)
            comp_type = 1
            layer_bytes = compressed

        f.write(layer_bytes)
        compressed_size = len(layer_bytes)

        # Build 48-byte layer index entry
        entry = struct.pack('<QIIBBHIIfIIQ',
            offset,                              # offset
            compressed_size,                     # compressed_size
            decompressed_size,                   # decompressed_size
            self.quant_type,                     # quant_type
            comp_type,                           # compression_type
            num_tensors,                         # num_tensors
            crc32(layer_bytes),                  # crc32_compressed
            crc32(bytes(layer_buf)),             # crc32_decompressed
            0.0,                                 # importance_score
            decompressed_size,                   # weight_bytes
            0,                                   # layer_ffn_dim (0=use global)
            0,                                   # reserved (8 bytes)
        )
        return entry[:48]  # ensure 48 bytes

    def _write_final(self, f, loader: TensorLoader,
                     mapping: Dict, arch: str, cfg: dict):
        """Write final section: norm weights + optional output head."""
        # Load final norm weights
        if arch == 'gpt2':
            norm_w = loader.get_tensor('transformer.ln_f.weight')
            norm_b = loader.get_tensor('transformer.ln_f.bias')
        elif arch == 'phi':
            norm_w = loader.get_tensor('model.final_layernorm.weight')
            norm_b = loader.get_tensor('model.final_layernorm.bias')
        else:
            norm_w = loader.get_tensor('model.norm.weight')
            norm_b = None

        if norm_w is None:
            raise RuntimeError("Final norm weights not found!")

        norm_data = norm_w.astype(np.float16).tobytes()
        if norm_b is not None:
            norm_data += norm_b.astype(np.float16).tobytes()

        # Output head
        tie = cfg.get('tie_word_embeddings', arch == 'gpt2')
        output_head_type = 1 if tie else 0
        output_head_data = b''
        if not tie:
            head_tensor = loader.get_tensor('lm_head.weight')
            if head_tensor is not None:
                output_head_data, _, _ = self.quantizer.quantize_tensor(head_tensor)
                del head_tensor
                head_bias = loader.get_tensor('lm_head.bias')
                if head_bias is not None:
                    output_head_data += head_bias.astype(np.float16).tobytes()
                    output_head_type = 2

        # Final section header (16 bytes)
        section_data = norm_data + output_head_data
        final_header = struct.pack('<IIII',
            output_head_type,
            len(norm_data),
            len(output_head_data),
            crc32(section_data),
        )

        f.write(final_header)
        f.write(section_data)


###############################################################################
# Main
###############################################################################
def parse_config(model_dir: Path) -> dict:
    """Parse config.json from model directory."""
    config_file = model_dir / 'config.json'
    if not config_file.exists():
        raise RuntimeError(f"config.json not found in {model_dir}")
    with open(config_file) as f:
        return json.load(f)


def detect_architecture(config: dict) -> str:
    """Detect model architecture from config."""
    model_type = config.get('model_type', '').lower()
    arch_aliases = {
        'gpt2': 'gpt2',
        'llama': 'llama',
        'mistral': 'mistral',
        'phi': 'phi', 'phi-msft': 'phi', 'phi3': 'phi',
        'gptj': 'gptj', 'gpt-j': 'gptj',
        'gpt_neox': 'gpt_neox',
        'qwen2': 'qwen2',
        'gemma': 'gemma', 'gemma2': 'gemma',
        'stablelm': 'stablelm',
    }
    arch = arch_aliases.get(model_type)
    if arch is None:
        log.warning(f"Unknown model_type '{model_type}', defaulting to llama")
        arch = 'llama'
    return arch


QUANT_TYPE_MAP = {
    '2bit': QUANT_2BIT_ASYM,
    '3bit': QUANT_3BIT_ASYM,
    '4bit': QUANT_4BIT_ASYM,
    '4bit_sym': QUANT_4BIT_SYM,
    'fp16': QUANT_FP16,
}


def main():
    parser = argparse.ArgumentParser(
        description='Convert HuggingFace model to QSF format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python convert_hf.py ./llama-7b/ -o llama-7b.qsf --quant 4bit
  python convert_hf.py ./gpt2/ -o gpt2.qsf --quant 4bit_sym
  python convert_hf.py ./mistral-7b/ -o mistral.qsf --quant 2bit --compress
""")
    parser.add_argument('model_dir', help='HuggingFace model directory')
    parser.add_argument('-o', '--output', required=True, help='Output .qsf file')
    parser.add_argument('--quant', choices=list(QUANT_TYPE_MAP.keys()),
                        default='4bit', help='Quantization type (default: 4bit)')
    parser.add_argument('--block-size', type=int, default=64,
                        choices=[32, 64, 128, 256],
                        help='Quantization block size (default: 64)')
    parser.add_argument('--compress', action='store_true',
                        help='LZ4-compress layer data')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        log.error(f"Not a directory: {model_dir}")
        sys.exit(1)

    # 1. Parse config
    log.info(f"Loading model from: {model_dir}")
    config = parse_config(model_dir)
    arch = detect_architecture(config)

    log.info(f"Architecture: {arch}")
    log.info(f"  Hidden dim:  {config['hidden_size']}")
    log.info(f"  Layers:      {config['num_hidden_layers']}")
    log.info(f"  Heads:       {config['num_attention_heads']}")
    log.info(f"  KV heads:    {config.get('num_key_value_heads', config['num_attention_heads'])}")
    log.info(f"  Vocab:       {config['vocab_size']}")
    log.info(f"  Intermediate:{config.get('intermediate_size', '?')}")

    # 2. Parse tokenizer
    tok_data = parse_tokenizer(model_dir)

    # 3. Index tensors
    loader = TensorLoader(args.model_dir)

    # 4. Convert
    quant_type = QUANT_TYPE_MAP[args.quant]
    writer = QSFWriter(args.output, config, quant_type,
                       block_size=args.block_size,
                       compress=args.compress)
    writer.write(loader, arch, tok_data)

    log.info("Conversion complete!")


if __name__ == '__main__':
    main()
