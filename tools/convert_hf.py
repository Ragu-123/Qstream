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

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


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
    'stablelm': 7, 'mixtral': 8, 'deepseek': 9, 'gpt_oss': 10,
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
FFN_MOE      = 3

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
TENSOR_MOE_ROUTER      = 21
TENSOR_MOE_ROUTER_BIAS = 22
TENSOR_MOE_GATE_0      = 23
TENSOR_MOE_GATE_BIAS_0 = 24
TENSOR_MOE_UP_0        = 25
TENSOR_MOE_UP_BIAS_0   = 26
TENSOR_MOE_DOWN_0      = 27
TENSOR_MOE_DOWN_BIAS_0 = 28

def moe_gate_id(e, bias=False):
    base = TENSOR_MOE_GATE_BIAS_0 if bias else TENSOR_MOE_GATE_0
    return base + e * 6

def moe_up_id(e, bias=False):
    base = TENSOR_MOE_UP_BIAS_0 if bias else TENSOR_MOE_UP_0
    return base + e * 6

def moe_down_id(e, bias=False):
    base = TENSOR_MOE_DOWN_BIAS_0 if bias else TENSOR_MOE_DOWN_0
    return base + e * 6


###############################################################################
# CRC-32 (matches format.c)
###############################################################################
def crc32(data: bytes) -> int:
    """Standard CRC-32 computation matching the C implementation."""
    import binascii
    return binascii.crc32(data) & 0xFFFFFFFF


###############################################################################
# Quantization — Vectorized with Outlier-Aware Support
###############################################################################

# New quant type for outlier-aware compression
QUANT_2BIT_OUTLIER = 9   # 2-bit + FP16 outliers
QUANT_4BIT_OUTLIER = 10  # 4-bit + FP16 outliers

class Quantizer:
    """Vectorized block quantizer for 2/3/4-bit quantization.

    Uses numpy vectorized operations — NO per-block Python loops.
    Supports outlier-aware mode: top outlier_frac weights stored as FP16,
    rest quantized at lower bits with tighter range for better accuracy.
    """

    def __init__(self, quant_type: int, block_size: int = 64,
                 outlier_frac: float = 0.0):
        self.quant_type = quant_type
        self.block_size = block_size
        self.outlier_frac = outlier_frac  # 0.0 = off, 0.005 = 0.5%

    def quantize_tensor(self, tensor: np.ndarray) -> Tuple[bytes, int, int]:
        """Quantize a 2D tensor. Returns (packed_bytes, rows, cols)."""
        if isinstance(tensor, np.ndarray):
            tensor = tensor.astype(np.float32, copy=False)
        else:
            # Handle torch tensors
            if hasattr(tensor, 'cpu'):
                tensor = tensor.float().cpu().numpy()
            else:
                tensor = np.asarray(tensor, dtype=np.float32)

        if tensor.ndim == 1:
            tensor = tensor.reshape(1, -1)
        rows, cols = tensor.shape

        if self.quant_type == QUANT_FP16:
            return tensor.astype(np.float16).tobytes(), rows, cols

        # Flatten for block processing
        flat = tensor.flatten()
        bs = self.block_size
        n = len(flat)

        # Pad to block boundary
        pad_len = (bs - (n % bs)) % bs
        if pad_len > 0:
            flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.float32)])

        # Reshape to blocks: [num_blocks, block_size]
        blocks = flat.reshape(-1, bs)

        if self.quant_type == QUANT_4BIT_ASYM:
            data = self._vec_quant_nbit(blocks, bits=4, symmetric=False)
        elif self.quant_type == QUANT_4BIT_SYM:
            data = self._vec_quant_nbit_sym(blocks, bits=4)
        elif self.quant_type == QUANT_2BIT_ASYM:
            data = self._vec_quant_nbit(blocks, bits=2, symmetric=False)
        elif self.quant_type == QUANT_3BIT_ASYM:
            data = self._vec_quant_3bit(blocks)
        elif self.quant_type in (QUANT_2BIT_OUTLIER, QUANT_4BIT_OUTLIER):
            data = self._vec_quant_outlier(flat[:n], rows, cols)
            return data, rows, cols
        else:
            raise ValueError(f"Unknown quant type: {self.quant_type}")

        return data, rows, cols

    def _vec_quant_nbit(self, blocks, bits, symmetric=False):
        """Vectorized N-bit asymmetric quantization (2-bit or 4-bit).

        Processes ALL blocks in parallel with numpy — no Python loops.

        Block format: [scale_fp16(2) + zero_fp16(2) + packed_data]
        """
        max_val = (1 << bits) - 1  # 3 for 2-bit, 15 for 4-bit
        nb, bs = blocks.shape

        # 1. Per-block min/max (vectorized)
        bmin = blocks.min(axis=1)
        bmax = blocks.max(axis=1)

        # 2. Scale
        rng = bmax - bmin
        scale = np.where(rng > 1e-30, rng / float(max_val), 0.0)
        inv_scale = np.where(scale > 1e-30, 1.0 / scale, 0.0)

        # 3. Quantize all blocks at once
        vals = (blocks - bmin[:, np.newaxis]) * inv_scale[:, np.newaxis]
        vals = np.clip(np.round(vals), 0, max_val).astype(np.uint8)

        # 4. Convert scale/min to FP16 bytes
        scale_fp16 = scale.astype(np.float16).view(np.uint8).reshape(nb, 2)
        min_fp16 = bmin.astype(np.float16).view(np.uint8).reshape(nb, 2)

        # 5. Pack bits
        if bits == 4:
            # Pack two 4-bit values per byte
            low = vals[:, 0::2]
            high = vals[:, 1::2]
            packed = (low | (high << 4)).astype(np.uint8)
        elif bits == 2:
            # Pack four 2-bit values per byte
            packed_len = (bs * 2 + 7) // 8
            packed = np.zeros((nb, packed_len), dtype=np.uint8)
            for k in range(4):
                idx = np.arange(k, bs, 4)
                byte_idx = idx // 4
                if len(idx) > 0:
                    packed[:, byte_idx] |= (vals[:, idx] & 0x03) << (k * 2)
        else:
            raise ValueError(f"Unsupported bit width: {bits}")

        # 6. Concatenate: [header(4) + packed_data] per block
        result = np.concatenate([scale_fp16, min_fp16, packed], axis=1)
        return result.tobytes()

    def _vec_quant_nbit_sym(self, blocks, bits):
        """Vectorized N-bit symmetric quantization."""
        max_val = (1 << (bits - 1)) - 1  # 7 for 4-bit
        nb, bs = blocks.shape

        absmax = np.abs(blocks).max(axis=1)
        scale = np.where(absmax > 1e-30, absmax / float(max_val), 0.0)
        inv_scale = np.where(scale > 1e-30, 1.0 / scale, 0.0)

        vals = blocks * inv_scale[:, np.newaxis]
        vals = np.clip(np.round(vals), -max_val, max_val).astype(np.int8)
        # Store as unsigned (offset by max_val+1)
        vals = (vals + max_val + 1).astype(np.uint8)

        scale_fp16 = scale.astype(np.float16).view(np.uint8).reshape(nb, 2)
        # 2 bytes padding for format compatibility
        padding = np.zeros((nb, 2), dtype=np.uint8)

        if bits == 4:
            low = vals[:, 0::2]
            high = vals[:, 1::2]
            packed = (low | (high << 4)).astype(np.uint8)
        else:
            raise ValueError("Symmetric only supports 4-bit currently")

        result = np.concatenate([scale_fp16, padding, packed], axis=1)
        return result.tobytes()

    def _vec_quant_3bit(self, blocks):
        """Vectorized 3-bit asymmetric quantization."""
        nb, bs = blocks.shape

        bmin = blocks.min(axis=1)
        bmax = blocks.max(axis=1)
        rng = bmax - bmin
        scale = np.where(rng > 1e-30, rng / 7.0, 0.0)
        inv_scale = np.where(scale > 1e-30, 1.0 / scale, 0.0)

        vals = (blocks - bmin[:, np.newaxis]) * inv_scale[:, np.newaxis]
        vals = np.clip(np.round(vals), 0, 7).astype(np.uint8)

        scale_fp16 = scale.astype(np.float16).view(np.uint8).reshape(nb, 2)
        min_fp16 = bmin.astype(np.float16).view(np.uint8).reshape(nb, 2)

        # Pack 3-bit values into bytes
        total_bits = bs * 3
        packed_bytes = (total_bits + 7) // 8
        packed = np.zeros((nb, packed_bytes), dtype=np.uint8)

        for i in range(bs):
            bit_pos = i * 3
            byte_idx = bit_pos // 8
            bit_off = bit_pos % 8
            packed[:, byte_idx] |= (vals[:, i] & 0x07) << bit_off
            if bit_off > 5:
                packed[:, byte_idx + 1] |= (vals[:, i] & 0x07) >> (8 - bit_off)

        result = np.concatenate([scale_fp16, min_fp16, packed], axis=1)
        return result.tobytes()

    def _vec_quant_outlier(self, flat, rows, cols):
        """Outlier-aware quantization: extract top outliers as FP16,
        quantize remainder at lower bits with tighter range.

        Format: [num_outliers(4)] [outlier_entries(6*N)] [quantized_blocks]
        Each outlier: [flat_index(4) + fp16_value(2)] = 6 bytes
        """
        n = len(flat)
        frac = self.outlier_frac if self.outlier_frac > 0 else 0.005  # default 0.5%
        num_outliers = max(1, int(n * frac))

        # Find top outliers by magnitude
        abs_vals = np.abs(flat)
        # Use argpartition for O(n) instead of O(n log n) sort
        outlier_indices = np.argpartition(abs_vals, -num_outliers)[-num_outliers:]
        outlier_values = flat[outlier_indices].astype(np.float16)

        # Zero out outliers in the tensor
        flat_clean = flat.copy()
        flat_clean[outlier_indices] = 0.0

        # Pack outlier entries: [index(u32) + value(fp16)] × N
        outlier_buf = bytearray(4)  # num_outliers header
        struct.pack_into('<I', outlier_buf, 0, num_outliers)
        for idx, val in zip(outlier_indices, outlier_values):
            outlier_buf += struct.pack('<IH', int(idx), val.view(np.uint16))

        # Quantize the cleaned tensor at the base bit width
        bs = self.block_size
        pad_len = (bs - (n % bs)) % bs
        if pad_len > 0:
            flat_clean = np.concatenate([flat_clean,
                                          np.zeros(pad_len, dtype=np.float32)])
        blocks = flat_clean.reshape(-1, bs)

        if self.quant_type == QUANT_2BIT_OUTLIER:
            quant_data = self._vec_quant_nbit(blocks, bits=2, symmetric=False)
        else:
            quant_data = self._vec_quant_nbit(blocks, bits=4, symmetric=False)

        return bytes(outlier_buf) + quant_data

    # Legacy method names for compatibility
    def _float_to_fp16(self, val: float) -> int:
        return int(np.float16(val).view(np.uint16))

    def block_data_size(self, num_elements: int) -> int:
        bs = self.block_size
        num_blocks = (num_elements + bs - 1) // bs
        if self.quant_type in (QUANT_4BIT_ASYM,):
            return num_blocks * (4 + bs // 2)
        elif self.quant_type == QUANT_4BIT_SYM:
            return num_blocks * (4 + bs // 2)
        elif self.quant_type == QUANT_2BIT_ASYM:
            return num_blocks * (4 + (bs * 2 + 7) // 8)
        elif self.quant_type == QUANT_3BIT_ASYM:
            return num_blocks * (4 + (bs * 3 + 7) // 8)
        elif self.quant_type == QUANT_FP16:
            return num_elements * 2
        return 0


###############################################################################
# Architecture-specific tensor name mapping
###############################################################################
def get_tensor_mapping(arch: str, num_layers: int,
                       config: dict = None) -> Dict[str, Tuple[str, int]]:
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


    # Generic MoE Support (Mixtral, DeepSeek, Qwen-MoE, GPT-OSS)
    # Check all known MoE patterns if the architecture is identified as MoE
    # or if we fell back to 'mixtral' during auto-detection.
    elif arch in ('mixtral', 'deepseek', 'gpt_oss', 'qwen2_moe'):
        # Global mappings (try standard names)
        mapping['model.embed_tokens.weight'] = ('embedding', -1)
        mapping['model.norm.weight'] = ('final_norm_w', -1)
        mapping['lm_head.weight'] = ('output_head', -1)
        # Some models use 'transformer.wte' etc. (checked via specific arch blocks above if needed)

        for L in range(num_layers):
            p = f'model.layers.{L}'
            
            # 1. Attention (Standard & Qwen/DeepSeek variants)
            # Standard LLaMA/Mistral names
            mapping[f'{p}.self_attn.q_proj.weight'] = ('attn_q_w', L)
            mapping[f'{p}.self_attn.k_proj.weight'] = ('attn_k_w', L)
            mapping[f'{p}.self_attn.v_proj.weight'] = ('attn_v_w', L)
            mapping[f'{p}.self_attn.o_proj.weight'] = ('attn_o_w', L)
            # Attention biases (GPT-OSS, Qwen, etc.)
            mapping[f'{p}.self_attn.q_proj.bias'] = ('attn_q_b', L)
            mapping[f'{p}.self_attn.k_proj.bias'] = ('attn_k_b', L)
            mapping[f'{p}.self_attn.v_proj.bias'] = ('attn_v_b', L)
            mapping[f'{p}.self_attn.o_proj.bias'] = ('attn_o_b', L)
            # Qwen/DeepSeek might use 'c_attn' (fused) or different names? 
            # Usually they follow LLaMA conventions in HF.

            # Norms
            mapping[f'{p}.input_layernorm.weight'] = ('attn_norm_w', L)
            mapping[f'{p}.post_attention_layernorm.weight'] = ('ffn_norm_w', L)
            # Qwen uses 'rms_norm' sometimes? standardizing on LLaMA names for now.

            # 2. MoE Router (Gate) - Try common names
            # Mixtral: block_sparse_moe.gate
            mapping[f'{p}.block_sparse_moe.gate.weight'] = ('moe_router_w', L)
            # DeepSeek/Qwen: mlp.gate
            mapping[f'{p}.mlp.gate.weight'] = ('moe_router_w', L)
            # GPT-OSS: mlp.router
            mapping[f'{p}.mlp.router.weight'] = ('moe_router_w', L)
            mapping[f'{p}.mlp.router.bias']   = ('moe_router_b', L)
            # GPT-OSS old?: block_sparse_layer.gate?
            mapping[f'{p}.block_sparse_layer.gate.weight'] = ('moe_router_w', L)

            # 3. Experts - Try common names
            num_experts = config.get('num_local_experts', 
                            config.get('n_routed_experts', 
                              config.get('num_experts', 8)))
            
            for e in range(num_experts):
                # Expert prefix patterns
                prefixes = [
                    f'{p}.block_sparse_moe.experts.{e}', # Mixtral
                    f'{p}.mlp.experts.{e}',               # DeepSeek/Qwen
                    f'{p}.block_sparse_layer.experts.{e}' # GPT-OSS
                ]
                
                for ep in prefixes:
                    # Pattern A: w1(gate), w3(up), w2(down) - Mixtral style
                    mapping[f'{ep}.w1.weight'] = (f'moe_gate_w_{e}', L)
                    mapping[f'{ep}.w3.weight'] = (f'moe_up_w_{e}', L)
                    mapping[f'{ep}.w2.weight'] = (f'moe_down_w_{e}', L)
                    # Biases (rare for Mixtral but possible)
                    mapping[f'{ep}.w1.bias'] = (f'moe_gate_b_{e}', L)
                    mapping[f'{ep}.w3.bias'] = (f'moe_up_b_{e}', L)
                    mapping[f'{ep}.w2.bias'] = (f'moe_down_b_{e}', L)
                    
                    # Pattern B: gate_proj, up_proj, down_proj - LLaMA style
                    mapping[f'{ep}.gate_proj.weight'] = (f'moe_gate_w_{e}', L)
                    mapping[f'{ep}.up_proj.weight']   = (f'moe_up_w_{e}', L)
                    mapping[f'{ep}.down_proj.weight'] = (f'moe_down_w_{e}', L)
                    # Biases (GPT-OSS uses these)
                    mapping[f'{ep}.gate_proj.bias'] = (f'moe_gate_b_{e}', L)
                    mapping[f'{ep}.up_proj.bias']   = (f'moe_up_b_{e}', L)
                    mapping[f'{ep}.down_proj.bias'] = (f'moe_down_b_{e}', L)


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

    def keys(self):
        """Yield all tensor names known to this loader."""
        return self.tensors.keys()

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
            if HAS_TORCH:
                # Use PyTorch backend to handle bfloat16 and preserve int types
                with safe_open(filepath, framework='pt') as st:
                    pt_tensor = st.get_tensor(key)
                    if pt_tensor.is_floating_point():
                        return pt_tensor.float().numpy()
                    else:
                        return pt_tensor.numpy()
            else:
                # Fallback to numpy
                with safe_open(filepath, framework='numpy') as st:
                    arr = st.get_tensor(key)
        else:
            sd = torch.load(filepath, map_location='cpu', weights_only=True)
            t = sd[key]
            if hasattr(t, 'is_floating_point') and t.is_floating_point():
               arr = t.float().numpy()
            elif hasattr(t, 'numpy'):
               arr = t.numpy()
            else:
               arr = t
            del sd

        # Convert float16/bfloat16 to float32 (if numpy)
        if hasattr(arr, 'dtype') and arr.dtype.kind == 'f':
             if arr.dtype == np.float16 or str(arr.dtype) == 'bfloat16':
                 arr = arr.astype(np.float32)

        return arr

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

    # Merge data: list of "tokenA tokenB" strings OR ["tokenA", "tokenB"] lists
    merge_buf = bytearray()
    for merge_str in tok_data['merges']:
        if isinstance(merge_str, list):
            parts = merge_str
        else:
            parts = merge_str.split(' ', 1)
        
        if len(parts) == 2:
            a = tok_data['vocab'].get(parts[0], 0)
            b = tok_data['vocab'].get(parts[1], 0)
            # Find merged token
            merged_str = parts[0] + parts[1]
            merged_id = tok_data['vocab'].get(merged_str, 0)
            merge_buf += struct.pack('<III', a, b, merged_id)

    num_merges = len(tok_data['merges'])

    # CRC32 of vocab data (required by C reader which skips 4 bytes after vocab)
    import binascii
    vocab_crc = binascii.crc32(bytes(vocab_buf)) & 0xFFFFFFFF

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

    # Format: [header][vocab_data][crc32][merge_data]
    # The C reader (tokenizer.c) expects a 4-byte CRC32 between vocab and merges
    return header + bytes(vocab_buf) + struct.pack('<I', vocab_crc) + bytes(merge_buf)


###############################################################################
# QSF File Writer
###############################################################################
class QSFWriter:
    """Writes QSF format files."""

    def __init__(self, output_path: str, config: dict, quant_type: int,
                 block_size: int = 64, compress: bool = False, device: str = 'cpu',
                 outlier_frac: float = 0.0):
        self.output_path = output_path
        self.config = config
        self.quant_type = quant_type
        self.block_size = block_size
        self.compress = compress and HAS_LZ4

        self.device = device
        if device.startswith('cuda') and HAS_TORCH:
            log.info(f"Using GPU quantization on {device}")
            self.quantizer = GPUQuantizer(quant_type, block_size, device)
        else:
            self.quantizer = Quantizer(quant_type, block_size,
                                        outlier_frac=outlier_frac)
        # Norms always use FP16 on CPU (they're tiny)
        self.norm_quantizer = Quantizer(QUANT_FP16, block_size)

    def write(self, loader: TensorLoader, arch: str, tok_data: dict):
        """Write complete QSF file."""
        cfg = self.config
        num_layers = cfg['num_hidden_layers']

        num_experts = cfg.get('num_local_experts',
                       cfg.get('n_routed_experts', 0))
        self.num_experts = num_experts
        self.config = cfg   # store for MoE access

        log.info(f"Writing QSF: {arch}, {num_layers} layers, "
                 f"quant={self.quant_type}, bs={self.block_size}")
        if num_experts > 1:
            nae = cfg.get('num_experts_per_tok', 2)
            log.info(f"  MoE: {num_experts} experts, top-{nae}")

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

            # ── 4. Write each layer (sequential + batch preload) ──
            mapping = get_tensor_mapping(arch, num_layers, cfg)
            for layer_idx in tqdm(range(num_layers), desc="Converting layers"):
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
        num_experts = cfg.get('num_local_experts',
                       cfg.get('n_routed_experts', 0))
        if num_experts > 1:
            ffn_type = FFN_MOE
        elif arch in ('llama', 'mistral', 'qwen2', 'gemma'):
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

        # Bias bitfield: detect from config
        has_bias = 0
        if arch == 'gpt2':
            has_bias = 0x0F  # all biases
        elif arch == 'phi':
            has_bias = 0x03  # QKV + output bias
        elif cfg.get('attention_bias', False):
            has_bias |= 0x01  # QKV bias
            has_bias |= 0x02  # output bias
        if cfg.get('mlp_bias', False) or (num_experts > 1 and arch == 'gpt_oss'):
            has_bias |= 0x04  # FFN/MoE expert bias

        # Weight tying
        tie = cfg.get('tie_word_embeddings', arch == 'gpt2')

        # RoPE scaling
        rope_scaling_type = 0
        rope_scaling_factor = 1.0
        rope_scaling_cfg = cfg.get('rope_scaling')
        if rope_scaling_cfg:
            rs_type = rope_scaling_cfg.get('type',
                      rope_scaling_cfg.get('rope_type', ''))
            if rs_type == 'linear':
                rope_scaling_type = 1
                rope_scaling_factor = rope_scaling_cfg.get('factor', 1.0)
            elif rs_type == 'ntk':
                rope_scaling_type = 2
            elif rs_type == 'yarn':
                rope_scaling_type = 3
                rope_scaling_factor = rope_scaling_cfg.get('factor', 1.0)
            elif rs_type == 'dynamic':
                rope_scaling_type = 4

        sliding_window = cfg.get('sliding_window', 0) or 0

        # Compute param count estimate
        intermediate_dim = cfg.get('intermediate_size', 4 * hidden_dim)
        if num_experts > 1:
            # MoE: attention + num_experts * expert_ffn + router
            expert_ffn_params = num_experts * 3 * hidden_dim * intermediate_dim
            attn_params = 4 * hidden_dim * (num_kv_heads * head_dim + hidden_dim)
            total_params = (cfg['vocab_size'] * hidden_dim +
                            cfg['num_hidden_layers'] * (attn_params + expert_ffn_params))
        else:
            total_params = (cfg['vocab_size'] * hidden_dim +
                            cfg['num_hidden_layers'] * (
                                4 * hidden_dim * hidden_dim +
                                2 * hidden_dim * intermediate_dim
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

        # MoE fields (offset 173-180)
        num_experts = cfg.get('num_local_experts',
                       cfg.get('n_routed_experts', 0))
        num_active = cfg.get('num_experts_per_tok', 2)
        moe_norm = 1 if num_experts > 1 else 0
        expert_ffn_dim = cfg.get('expert_intermediate_size',
                           cfg.get('intermediate_size', 4 * hidden_dim))
        struct.pack_into('<B', buf, 173, min(num_experts, 255))
        struct.pack_into('<B', buf, 174, min(num_active, 255))
        struct.pack_into('<B', buf, 175, moe_norm)
        # byte 176 is reserved padding (already zero)
        struct.pack_into('<I', buf, 177, expert_ffn_dim)

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
        # Batch-preload all expert tensors for this layer
        if hasattr(loader, 'preload_layer'):
            loader.preload_layer(layer_idx)
        
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
            'moe_router_w': TENSOR_MOE_ROUTER,
        }
        # Add per-expert MoE tensor IDs dynamically
        num_experts = cfg.get('num_local_experts',
                       cfg.get('n_routed_experts', 0))
        # Router
        role_to_type['moe_router_w'] = TENSOR_MOE_ROUTER
        role_to_type['moe_router_b'] = TENSOR_MOE_ROUTER_BIAS
        
        for e in range(num_experts):
            # Weights
            role_to_type[f'moe_gate_w_{e}'] = moe_gate_id(e)
            role_to_type[f'moe_up_w_{e}']   = moe_up_id(e)
            role_to_type[f'moe_down_w_{e}'] = moe_down_id(e)
            # Biases
            role_to_type[f'moe_gate_b_{e}'] = moe_gate_id(e, bias=True)
            role_to_type[f'moe_up_b_{e}']   = moe_up_id(e, bias=True)
            role_to_type[f'moe_down_b_{e}'] = moe_down_id(e, bias=True)

        num_tensors = 0
        layer_buf = bytearray()
        written_roles = set()  # Prevent duplicate tensor writes (MoE has multiple HF name aliases)

        for hf_name, (role, lidx) in mapping.items():
            if lidx != layer_idx:
                continue

            # Skip if we already wrote this role for this layer
            if role in written_roles:
                continue

            tensor_type_id = role_to_type.get(role)
            if tensor_type_id is None:
                continue

            tensor = loader.get_tensor(hf_name)
            if tensor is None:
                continue

            written_roles.add(role)

            # GPT-2 conv1d weights are transposed
            if arch == 'gpt2' and role in ('fused_qkv_w', 'attn_o_w',
                                            'ffn_up_w', 'ffn_down_w'):
                tensor = tensor.T.copy()

            # Choose quantizer: norms and biases stay in FP16, weights get quantized
            if role in ('attn_norm_w', 'attn_norm_b', 'ffn_norm_w', 'ffn_norm_b') or \
               'bias' in role or role.endswith('_b'):
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
        # Free batch cache after processing layer
        if hasattr(loader, 'clear_batch_cache'):
            loader.clear_batch_cache()
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
# GPT-OSS Loader (MXFP4 handling)
###############################################################################
class GptOssLoader(TensorLoader):
    """Loader for GPT-OSS models with MXFP4 quantized stacked experts.
    
    Uses batch-load-then-cache: loads stacked tensors ONCE per layer,
    batch-dequants ALL 32 experts in one GPU operation, caches results.
    Reduces file opens from ~4600 to ~144 and GPU ops from ~2300 to ~48.
    """
    def __init__(self, model_dir, num_experts, device='cpu'):
        super().__init__(model_dir)
        self.num_experts = num_experts
        self.device = device
        if HAS_TORCH and device.startswith('cuda'):
            self.device_obj = torch.device(device)
        else:
            self.device_obj = None
        self._batch_cache = {}  # {('gate', expert_idx): tensor, ...}

    def _load_full_tensor_pt(self, name):
        """Load full tensor as PyTorch tensor preserving original dtype."""
        if name not in self.tensors:
            return None
        fpath, key = self.tensors[name]
        if not HAS_SAFETENSORS:
            return None
        with safe_open(fpath, framework='pt', device='cpu') as f:
            return f.get_tensor(key)

    def preload_layer(self, layer_idx):
        """Batch-load and dequant ALL experts for a layer in one GPU shot."""
        self._batch_cache.clear()
        prefix = f'model.layers.{layer_idx}.mlp.experts'

        # ── Batch-load weights ──
        for proj in ['gate_up_proj', 'down_proj']:
            blocks = self._load_full_tensor_pt(f'{prefix}.{proj}_blocks')
            scales = self._load_full_tensor_pt(f'{prefix}.{proj}_scales')
            if blocks is None or scales is None:
                continue

            # blocks: [E, R, B, 16]  scales: [E, R, B]
            E, R, B, sixteen = blocks.shape

            # Flatten expert+row dims for batch dequant
            blocks_flat = blocks.reshape(E * R, B, sixteen)
            scales_flat = scales.reshape(E * R, B)
            del blocks, scales  # free CPU copies

            # Move to GPU
            if self.device_obj:
                blocks_flat = blocks_flat.to(self.device_obj)
                scales_flat = scales_flat.float().to(self.device_obj)
            else:
                # CPU path: ensure float32 scales
                scales_flat = scales_flat.float()

            # Batch dequant ALL experts at once (1 GPU kernel launch)
            dequanted = self._dequant_mxfp4(blocks_flat, scales_flat)
            del blocks_flat, scales_flat

            # Reshape: [E*R, COLS] -> [E, R, COLS]
            COLS = dequanted.shape[1]
            dequanted = dequanted.reshape(E, R, COLS)

            # Split and cache individual experts (move to CPU to save GPU memory)
            is_gate_up = 'gate_up' in proj
            for e in range(E):
                expert_data = dequanted[e].float().cpu().numpy()  # [R, COLS]
                if is_gate_up:
                    split = R // 2
                    self._batch_cache[('gate', e)] = expert_data[:split]
                    self._batch_cache[('up', e)] = expert_data[split:]
                else:
                    self._batch_cache[('down', e)] = expert_data
            del dequanted

        # ── Batch-load biases (tiny, fast) ──
        for proj, is_gu in [('gate_up_proj_bias', True), ('down_proj_bias', False)]:
            bias = self._load_full_tensor_pt(f'{prefix}.{proj}')
            if bias is None:
                continue
            bias = bias.float()  # BF16 -> F32
            for e in range(bias.shape[0]):
                eb = bias[e].numpy()
                if is_gu:
                    split = eb.shape[0] // 2
                    self._batch_cache[('gate_bias', e)] = eb[:split]
                    self._batch_cache[('up_bias', e)] = eb[split:]
                else:
                    self._batch_cache[('down_bias', e)] = eb
            del bias

        log.debug(f"  Preloaded layer {layer_idx}: {len(self._batch_cache)} cached tensors")

    def clear_batch_cache(self):
        """Free all cached tensors and GPU memory."""
        self._batch_cache.clear()
        if HAS_TORCH and self.device_obj:
            torch.cuda.empty_cache()

    def keys(self):
        # Yield physical keys AND virtual keys for experts
        for name in super().keys():
            yield name
            if 'mlp.experts.gate_up_proj_blocks' in name:
                try:
                    parts = name.split('.')
                    l_idx = int(parts[2])
                    prefix = f'model.layers.{l_idx}.block_sparse_layer.experts'
                    for e in range(self.num_experts):
                        ep = f'{prefix}.{e}'
                        yield f'{ep}.gate.weight'
                        yield f'{ep}.up.weight'
                except:
                    pass
            if 'mlp.experts.down_proj_blocks' in name:
                try:
                    parts = name.split('.')
                    l_idx = int(parts[2])
                    prefix = f'model.layers.{l_idx}.block_sparse_layer.experts'
                    for e in range(self.num_experts):
                        yield f'{prefix}.{e}.down_proj.weight'
                except:
                    pass
            if 'mlp.experts.gate_up_proj_bias' in name:
                try:
                    parts = name.split('.')
                    l_idx = int(parts[2])
                    prefix = f'model.layers.{l_idx}.mlp.experts'
                    for e in range(self.num_experts):
                        ep = f'{prefix}.{e}'
                        yield f'{ep}.gate.bias'
                        yield f'{ep}.up.bias'
                except:
                    pass
            if 'mlp.experts.down_proj_bias' in name:
                try:
                    parts = name.split('.')
                    l_idx = int(parts[2])
                    prefix = f'model.layers.{l_idx}.mlp.experts'
                    for e in range(self.num_experts):
                        yield f'{prefix}.{e}.down_proj.bias'
                except:
                    pass

    def get_tensor(self, name):
        if 'experts' in name:
            if 'weight' in name:
                return self._get_expert_weight(name)
            elif 'bias' in name:
                return self._get_expert_bias(name)
        return super().get_tensor(name)

    def _get_expert_weight(self, name):
        """Return cached expert weight (already dequanted by preload_layer)."""
        try:
            parts = name.split('.')
            e_idx = int(parts[5])
            role = parts[6]
        except:
            return super().get_tensor(name)

        if role in ('gate', 'w1', 'gate_proj'):
            key = ('gate', e_idx)
        elif role in ('up', 'w3', 'up_proj'):
            key = ('up', e_idx)
        elif role in ('down', 'w2', 'down_proj'):
            key = ('down', e_idx)
        else:
            return super().get_tensor(name)

        return self._batch_cache.get(key)

    def _get_expert_bias(self, name):
        """Return cached expert bias."""
        try:
            parts = name.split('.')
            e_idx = int(parts[5])
            role = parts[6]
        except:
            return super().get_tensor(name)

        if role in ('gate', 'w1', 'gate_proj'):
            key = ('gate_bias', e_idx)
        elif role in ('up', 'w3', 'up_proj'):
            key = ('up_bias', e_idx)
        elif role in ('down', 'w2', 'down_proj'):
            key = ('down_bias', e_idx)
        else:
            return super().get_tensor(name)

        return self._batch_cache.get(key)

    def _dequant_mxfp4(self, blocks, scales):
        # blocks: [ROWS, N_BLOCKS, 16], uint8
        # scales: [ROWS, N_BLOCKS], float
        
        # Check for GPU execution
        if HAS_TORCH and isinstance(blocks, torch.Tensor):
             return self._dequant_mxfp4_torch(blocks, scales)
        
        rows, n_blocks, _ = blocks.shape
        
        # 1. Unpack uint8 -> two 4-bit indices
        # blocks is uint8 (guaranteed by get_tensor fix)
        lower = blocks & 0x0F
        upper = (blocks >> 4) & 0x0F
        
        # 2. Lookup values directly (E2M1)
        # Avoid creating intermediate large float/int arrays
        mxfp4_lut = np.array([
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,   # 0-7
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 # 8-15
        ], dtype=np.float32)

        # Lookup: uint8 -> float32 [ROWS, N_BLOCKS, 16]
        val_low  = mxfp4_lut[lower]
        val_high = mxfp4_lut[upper]
        
        # 3. Apply scales
        # scales: [ROWS, N_BLOCKS] -> [ROWS, N_BLOCKS, 1]
        s_bd = scales[:, :, np.newaxis]
        val_low  *= s_bd
        val_high *= s_bd
        
        # 4. Interleave into result
        # [ROWS, N_BLOCKS, 32]
        result = np.empty((rows, n_blocks, 32), dtype=np.float32)
        result[:, :, 0::2] = val_low
        result[:, :, 1::2] = val_high
        
        # Reshape to [ROWS, COLS]
        return result.reshape(rows, -1)

    def _dequant_mxfp4_torch(self, blocks, scales):
        """GPU/PyTorch implementation of MXFP4 dequantization."""
        rows, n_blocks, _ = blocks.shape
        
        # 1. Unpack uint8 -> two 4-bit indices
        lower = blocks & 0x0F
        upper = (blocks >> 4) & 0x0F
        
        # 2. Lookup values
        # Create LUT on same device
        mxfp4_lut = torch.tensor([
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,   # 0-7
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 # 8-15
        ], dtype=torch.float32, device=blocks.device)
        
        val_low = mxfp4_lut[lower.long()]
        val_high = mxfp4_lut[upper.long()]
        
        # 3. Apply scales
        # scales: [ROWS, N_BLOCKS] -> [ROWS, N_BLOCKS, 1]
        if isinstance(scales, np.ndarray): # ensure torch
             scales = torch.from_numpy(scales).to(blocks.device)
             
        s_bd = scales.unsqueeze(-1)
        val_low  *= s_bd
        val_high *= s_bd
        
        # 4. Interleave
        # [ROWS, N_BLOCKS, 32]
        result = torch.empty((rows, n_blocks, 32), dtype=torch.float32, device=blocks.device)
        result[:, :, 0::2] = val_low
        result[:, :, 1::2] = val_high
        
        return result.reshape(rows, -1)
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
        'mixtral': 'mixtral',
        'phi': 'phi', 'phi-msft': 'phi', 'phi3': 'phi',
        'gptj': 'gptj', 'gpt-j': 'gptj',
        'gpt_neox': 'gpt_neox',
        'qwen2': 'qwen2',
        'gemma': 'gemma', 'gemma2': 'gemma',
        'stablelm': 'stablelm',
        'deepseek': 'deepseek', 'deepseek_v2': 'deepseek',
        'gpt_oss': 'gpt_oss', 'qwen2_moe': 'qwen2_moe',
    }
    arch = arch_aliases.get(model_type)
    if arch is None:
        # Auto-detect MoE by checking for expert-related config keys
        if (config.get('num_local_experts') or 
            config.get('n_routed_experts') or 
            config.get('num_experts')):
            log.warning(f"Unknown model_type '{model_type}' with MoE config, "
                        f"defaulting to generic mixtral/moe path")
            arch = 'mixtral'
        else:
            log.warning(f"Unknown model_type '{model_type}', defaulting to llama")
            arch = 'llama'
    return arch


QUANT_TYPE_MAP = {
    '2bit': QUANT_2BIT_ASYM,
    '3bit': QUANT_3BIT_ASYM,
    '4bit': QUANT_4BIT_ASYM,
    '4bit_sym': QUANT_4BIT_SYM,
    '2bit_outlier': QUANT_2BIT_OUTLIER,
    '4bit_outlier': QUANT_4BIT_OUTLIER,
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
    parser.add_argument('--outlier-frac', type=float, default=0.0,
                        help='Outlier fraction (0.005 = 0.5%%). Extract top outliers as FP16.')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--device', default='cuda' if HAS_TORCH and torch.cuda.is_available() else 'cpu',
                        help='Device for quantization (cpu, cuda, cuda:0)')

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
    if arch == 'gpt_oss':
        num_experts = config.get('num_local_experts', 32)
        loader = GptOssLoader(args.model_dir, num_experts, device=args.device)
    else:
        loader = TensorLoader(args.model_dir)

    # 4. Convert
    quant_type = QUANT_TYPE_MAP[args.quant]
    writer = QSFWriter(args.output, config, quant_type,
                       block_size=args.block_size,
                       compress=args.compress,
                       device=args.device,
                       outlier_frac=args.outlier_frac)
    writer.write(loader, arch, tok_data)

    log.info("Conversion complete!")



###############################################################################
# GPU Quantizer (PyTorch-based)
###############################################################################
class GPUQuantizer:
    """Accelerated quantization using PyTorch on GPU.

    Supports all quant types on GPU (4-bit, 2-bit, 3-bit, fp16) instead of
    falling back to CPU for non-4-bit types.
    """
    def __init__(self, quant_type: int, block_size: int = 64, device='cuda'):
        self.quant_type = quant_type
        self.block_size = block_size
        self.device = device
        if HAS_TORCH:
            self.device = torch.device(device)

    def _to_gpu(self, tensor):
        """Move tensor to GPU, handling numpy arrays."""
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        return tensor.to(self.device)

    def quantize_tensor(self, tensor) -> Tuple[bytes, int, int]:
        """Quantize a tensor on GPU."""
        if not HAS_TORCH:
            raise RuntimeError("GPU quantization requires PyTorch")

        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        rows, cols = tensor.shape

        # Move to GPU
        tensor = self._to_gpu(tensor)

        if self.quant_type == QUANT_4BIT_ASYM:
            return self._quant_nbit_gpu(tensor, 4, symmetric=False), rows, cols
        elif self.quant_type == QUANT_4BIT_SYM:
            return self._quant_4bit_sym_gpu(tensor), rows, cols
        elif self.quant_type == QUANT_2BIT_ASYM:
            return self._quant_nbit_gpu(tensor, 2, symmetric=False), rows, cols
        elif self.quant_type == QUANT_3BIT_ASYM:
            return self._quant_3bit_gpu(tensor), rows, cols
        elif self.quant_type == QUANT_FP16:
            return tensor.half().cpu().numpy().tobytes(), rows, cols
        else:
            # Fallback for unknown types
            cpu_tensor = tensor.float().cpu().numpy()
            q = Quantizer(self.quant_type, self.block_size)
            return q.quantize_tensor(cpu_tensor)

    def _reshape_to_blocks(self, tensor):
        """Reshape flat tensor to [N_BLOCKS, BLOCK_SIZE], padding if needed."""
        numel = tensor.numel()
        bs = self.block_size
        pad_len = (bs - (numel % bs)) % bs
        if pad_len > 0:
            tensor = torch.nn.functional.pad(
                tensor.reshape(-1), (0, pad_len)).reshape(-1, bs)
        else:
            tensor = tensor.reshape(-1, bs)
        return tensor

    def _quant_nbit_gpu(self, tensor, bits, symmetric=False):
        """GPU N-bit asymmetric quantization (2-bit or 4-bit).

        Block format: [scale_fp16(2) + zero_fp16(2) + packed_data]
        """
        max_val = (1 << bits) - 1  # 3 for 2-bit, 15 for 4-bit
        tensor = self._reshape_to_blocks(tensor)
        bs = tensor.shape[1]

        # 1. Min/Max per block
        bmin = tensor.min(dim=1).values
        bmax = tensor.max(dim=1).values

        # 2. Scale
        scale = (bmax - bmin) / float(max_val)
        mask = (scale == 0)
        scale[mask] = 1.0
        inv_scale = 1.0 / scale
        inv_scale[mask] = 0.0

        # 3. Quantize all values
        vals = (tensor - bmin.unsqueeze(1)) * inv_scale.unsqueeze(1)
        vals = vals.round_().clamp_(0, max_val).to(torch.uint8)

        # 4. Pack bits and build per-block output
        scale_bytes = scale.to(torch.float16).contiguous().view(torch.uint8)
        min_bytes = bmin.to(torch.float16).contiguous().view(torch.uint8)

        if bits == 4:
            # Pack two 4-bit values per byte
            low = vals[:, 0::2]
            high = vals[:, 1::2]
            packed = (low | (high << 4))
            final = torch.cat([scale_bytes, min_bytes, packed], dim=1)
        elif bits == 2:
            # Pack four 2-bit values per byte
            n_blocks = vals.shape[0]
            packed_len = (bs * 2 + 7) // 8
            packed = torch.zeros(n_blocks, packed_len, dtype=torch.uint8,
                                 device=tensor.device)
            for k in range(4):
                idx = torch.arange(k, bs, 4, device=tensor.device)
                if idx.numel() > 0:
                    byte_idx = idx // 4
                    packed[:, byte_idx] |= (vals[:, idx] & 0x03) << (k * 2)
            final = torch.cat([scale_bytes, min_bytes, packed], dim=1)
        else:
            raise ValueError(f"Unsupported bit width: {bits}")

        return final.cpu().numpy().tobytes()

    def _quant_4bit_sym_gpu(self, tensor):
        """GPU 4-bit symmetric quantization.

        Block format: [scale_fp16(2) + packed_data]
        Values stored as unsigned 0..15 (offset by 8 from signed -7..+7)
        """
        tensor = self._reshape_to_blocks(tensor)
        bs = tensor.shape[1]

        absmax = tensor.abs().max(dim=1).values
        scale = absmax / 7.0
        mask = (scale == 0)
        scale[mask] = 1.0
        inv_scale = 1.0 / scale
        inv_scale[mask] = 0.0

        # Quantize to -7..+7, store as unsigned 0..15
        vals = (tensor * inv_scale.unsqueeze(1)).round_().clamp_(-7, 7)
        vals = (vals + 8).to(torch.uint8)

        scale_bytes = scale.to(torch.float16).contiguous().view(torch.uint8)
        # 2 bytes padding to match CPU format (skip scale + 2 reserved)
        padding = torch.zeros(scale_bytes.shape[0], 2, dtype=torch.uint8,
                              device=tensor.device)

        low = vals[:, 0::2]
        high = vals[:, 1::2]
        packed = (low | (high << 4))

        final = torch.cat([scale_bytes, padding, packed], dim=1)
        return final.cpu().numpy().tobytes()

    def _quant_3bit_gpu(self, tensor):
        """GPU 3-bit asymmetric quantization.

        Block format: [scale_fp16(2) + zero_fp16(2) + packed_3bit_data]
        """
        tensor = self._reshape_to_blocks(tensor)
        bs = tensor.shape[1]

        bmin = tensor.min(dim=1).values
        bmax = tensor.max(dim=1).values

        scale = (bmax - bmin) / 7.0
        mask = (scale == 0)
        scale[mask] = 1.0
        inv_scale = 1.0 / scale
        inv_scale[mask] = 0.0

        vals = (tensor - bmin.unsqueeze(1)) * inv_scale.unsqueeze(1)
        vals = vals.round_().clamp_(0, 7).to(torch.uint8)

        # Pack 3-bit values into bytes on CPU (bit-packing is tricky on GPU)
        vals_cpu = vals.cpu().numpy()
        n_blocks = vals_cpu.shape[0]
        total_bits = bs * 3
        packed_bytes = (total_bits + 7) // 8
        packed = np.zeros((n_blocks, packed_bytes), dtype=np.uint8)

        for i in range(bs):
            bit_pos = i * 3
            byte_idx = bit_pos // 8
            bit_off = bit_pos % 8
            packed[:, byte_idx] |= (vals_cpu[:, i] & 0x07) << bit_off
            if bit_off > 5:
                packed[:, byte_idx + 1] |= (vals_cpu[:, i] & 0x07) >> (8 - bit_off)

        scale_bytes = scale.to(torch.float16).cpu().contiguous().view(torch.uint8).numpy()
        min_bytes = bmin.to(torch.float16).cpu().contiguous().view(torch.uint8).numpy()

        # Build final: [scale(2) + min(2) + packed_data] per block
        final = np.concatenate([scale_bytes, min_bytes, packed], axis=1)
        return final.tobytes()


if __name__ == '__main__':
    main()
