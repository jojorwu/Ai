"""
PyTorch implementation of Multi-Head Attention with Grouped-Query Attention (GQA),
Rotary Positional Embeddings (RoPE), and Query-Key Normalization (QK Norm).
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

import torch
from torch import nn

from src.config.model_config import MultiHeadAttentionConfig
from src.model.layers.attention.attention import (
    AttentionInput,
    ScaledDotProductAttention,
)
from src.model.layers.core.linear import Linear
from src.model.layers.core.rms_norm import RMSNorm
from src.model.layers.attention.rotary_embedding import apply_rope_embeddings

if TYPE_CHECKING:
    from src.model.layers.attention.kv_cache import KVCache


class MultiHeadAttention(nn.Module):
    """
    Implements Grouped-Query Attention (GQA) with RoPE and QK Norm.

    GQA allows for faster inference and smaller KV cache by sharing Key/Value
    heads among multiple Query heads. QK Norm applies normalization to heads
    to improve training stability.
    """

    def __init__(
        self, config: MultiHeadAttentionConfig, linear_class: nn.Module = Linear
    ) -> None:
        """
        Initializes the MultiHeadAttention layer.

        Args:
            config: Configuration for the attention layer.
            linear_class: The linear layer class to use.
        """
        super().__init__()
        self._validate_config(config)
        self.config = config

        # Store frequently used attributes as instance variables
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.n_rep = config.num_heads // config.num_kv_heads
        self.head_dim = config.d_model // config.num_heads
        self.q_dim = self.head_dim * self.num_heads
        self.kv_dim = self.head_dim * self.num_kv_heads

        self.attention = ScaledDotProductAttention()
        self.qkv_proj, self.wo = self._create_projections(config, linear_class)

        # QK Norm: normalize heads separately to improve stability
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def _validate_config(self, config: MultiHeadAttentionConfig) -> None:
        """Validates that dimensions are compatible with head counts."""
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if config.num_heads % config.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")

    def _create_projections(
        self, config: MultiHeadAttentionConfig, linear_class: nn.Module
    ) -> Tuple[nn.Module, nn.Module]:
        """Creates Query, Key, Value and Output linear projections."""
        d_k = config.d_model // config.num_heads
        q_dim = d_k * config.num_heads
        kv_dim = d_k * config.num_kv_heads
        qkv_proj = linear_class(config.d_model, q_dim + 2 * kv_dim, bias=config.bias)
        if hasattr(qkv_proj, "special_residual_init"):
            qkv_proj.special_residual_init(config.num_layers)
        wo = linear_class(config.d_model, config.d_model, bias=config.bias)
        if hasattr(wo, "special_residual_init"):
            wo.special_residual_init(config.num_layers)
        return qkv_proj, wo

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Splits the last dimension into (heads, d_k) and transposes."""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merges the head and d_k dimensions back."""
        batch_size, _, seq_len, _ = x.shape
        # Performance: Use reshape() to automatically handle contiguity if needed.
        return x.transpose(1, 2).reshape(batch_size, seq_len, self.config.d_model)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeats KV heads to match Q heads for GQA."""
        if n_rep == 1:
            return x
        batch, num_kv_heads, seq_len, head_dim = x.shape
        return (
            x.unsqueeze(2)
            .expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
            .reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
        )

    def _prepare_qkv(
        self, x: torch.Tensor, kv_cache: KVCache | None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Prepares the query, key, and value projections."""
        seq_len = x.shape[1]
        seq_offset = kv_cache.current_pos if kv_cache is not None else 0
        qkv = self.qkv_proj(x)
        q_proj, k_proj, v_proj = qkv.split(
            [self.q_dim, self.kv_dim, self.kv_dim], dim=-1
        )
        q_proj = self._split_heads(q_proj, self.num_heads)
        k_proj = self._split_heads(k_proj, self.num_kv_heads)
        v_proj = self._split_heads(v_proj, self.num_kv_heads)

        # Apply QK Norm
        q_proj = self.q_norm(q_proj)
        k_proj = self.k_norm(k_proj)

        return q_proj, k_proj, v_proj, seq_len, seq_offset

    def _apply_rope(
        self, q: torch.Tensor, k: torch.Tensor, seq_offset: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies Rotary Positional Embeddings to Q and K."""
        if self.config.rotary_emb is not None:
            cos, sin = self.config.rotary_emb
            q = apply_rope_embeddings(q, cos, sin, seq_offset)
            k = apply_rope_embeddings(k, cos, sin, seq_offset)
        return q, k

    def _get_updated_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: KVCache | None,
        layer_idx: int | None,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Updates and retrieves K and V from the cache."""
        if kv_cache is not None:
            if layer_idx is None:
                raise ValueError("layer_idx must be provided when using kv_cache")
            kv_cache.update(k, v, layer_idx)
            k, v = kv_cache.get(layer_idx, seq_len=seq_len)
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the GQA layer.

        Args:
            x: Input hidden states [batch, seq_len, d_model].
            kv_cache: Optional persistent Key-Value cache.
            layer_idx: Index of the current layer (required if kv_cache is used).

        Returns:
            The output tensor after attention and projection.
        """
        seq_len = x.shape[1]
        q, k, v, _, seq_offset = self._prepare_qkv(x, kv_cache)
        q, k = self._apply_rope(q, k, seq_offset)
        k, v = self._get_updated_kv(k, v, kv_cache, layer_idx, seq_len)

        if self.n_rep > 1:
            k = self._repeat_kv(k, self.n_rep)
            v = self._repeat_kv(v, self.n_rep)

        # Causal masking is required when seq_len > 1 (e.g. prompt or speculative chunk).
        is_causal = (kv_cache is None) or (seq_len > 1)

        attn_input = AttentionInput(q=q, k=k, v=v, is_causal=is_causal)
        attention_output = self.attention(attn_input)
        combined_output = self._combine_heads(attention_output)
        return self.wo(combined_output)
