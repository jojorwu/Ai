"""
PyTorch implementation of Multi-Head Attention with Grouped-Query Attention (GQA)
and Rotary Positional Embeddings (RoPE).
"""
import torch
from torch import nn

from src.config import MultiHeadAttentionConfig
from src.nn_components.attention import ScaledDotProductAttention
from src.nn_components.linear import Linear
from src.nn_components.rotary_embedding import apply_rope_embeddings


class MultiHeadAttention(nn.Module):
    """
    Implements Grouped-Query Attention (GQA) with RoPE, migrated to PyTorch.
    """
    def __init__(self, config: MultiHeadAttentionConfig, linear_class=Linear):
        super().__init__()
        self._validate_config(config)
        self.config = config
        self.attention = ScaledDotProductAttention()
        self.qkv_proj, self.wo = self._create_projections(config, linear_class)

    def _validate_config(self, config):
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if config.num_heads % config.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")

    def _create_projections(self, config, linear_class):
        d_k = config.d_model // config.num_heads
        q_dim = d_k * config.num_heads
        kv_dim = d_k * config.num_kv_heads
        qkv_proj = linear_class(
            config.d_model, q_dim + 2 * kv_dim, bias=config.bias
        )
        if hasattr(qkv_proj, 'special_residual_init'):
            qkv_proj.special_residual_init(config.num_layers)
        wo = linear_class(config.d_model, config.d_model, bias=config.bias)
        if hasattr(wo, 'special_residual_init'):
            wo.special_residual_init(config.num_layers)
        return qkv_proj, wo

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Splits the last dimension into (heads, d_k) and transposes."""
        batch_size, seq_len, _ = x.shape
        d_k = self.config.d_model // self.config.num_heads
        return x.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merges the head and d_k dimensions back."""
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(
            1, 2
        ).contiguous().view(batch_size, seq_len, self.config.d_model)

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

    def _prepare_qkv(self, x, kv_cache):
        """Prepares the query, key, and value projections."""
        seq_len = x.shape[1]
        seq_offset = kv_cache.current_pos if kv_cache is not None else 0
        qkv = self.qkv_proj(x)
        d_k = self.config.d_model // self.config.num_heads
        q_dim = d_k * self.config.num_heads
        kv_dim = d_k * self.config.num_kv_heads
        q_proj, k_proj, v_proj = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        q_proj = self._split_heads(q_proj, self.config.num_heads)
        k_proj = self._split_heads(k_proj, self.config.num_kv_heads)
        v_proj = self._split_heads(v_proj, self.config.num_kv_heads)
        return q_proj, k_proj, v_proj, seq_len, seq_offset

    def forward(self, x: torch.Tensor, mask=None, kv_cache=None, layer_idx=None):
        """Forward pass of the GQA layer."""
        q_proj, k_proj, v_proj, seq_len, seq_offset = self._prepare_qkv(
            x, kv_cache
        )

        if self.config.rotary_emb is not None:
            cos, sin = self.config.rotary_emb
            q_proj = apply_rope_embeddings(q_proj, cos, sin, seq_offset)
            k_proj = apply_rope_embeddings(k_proj, cos, sin, seq_offset)

        if kv_cache is not None:
            kv_cache.update(k_proj, v_proj, layer_idx)
            k_proj, v_proj = kv_cache.get(layer_idx)
            mask = torch.triu(
                torch.ones(seq_len, k_proj.size(2)), diagonal=1
            ).bool().to(x.device)

        num_q_per_kv = self.config.num_heads // self.config.num_kv_heads
        k_proj = self._repeat_kv(k_proj, num_q_per_kv)
        v_proj = self._repeat_kv(v_proj, num_q_per_kv)

        attention_output = self.attention(q_proj, k_proj, v_proj, mask)
        combined_output = self._combine_heads(attention_output)
        return self.wo(combined_output)
