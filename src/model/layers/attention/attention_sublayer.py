"""
PyTorch implementation of the Attention Sub-Layer for a Transformer Decoder Block.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch
from torch import nn

from src.config.model_config import DecoderBlockConfig, MultiHeadAttentionConfig
from src.model.layers.core.dropout import Dropout
from src.model.layers.attention.multi_head_attention import MultiHeadAttention
from src.model.layers.core.rms_norm import RMSNorm
from src.model.layers.core.film import FiLMLayer

if TYPE_CHECKING:
    from src.model.layers.attention.kv_cache import KVCache


@dataclass
class AttentionSubLayerInput:
    """
    Inputs for the AttentionSubLayer.

    Attributes:
        x: Input tensor of shape [batch, seq_len, d_model].
        ltm_state: Long-Term Memory state tensor.
        kv_cache: Persistent Key-Value cache.
        layer_idx: Index of the current layer.
    """
    x: torch.Tensor
    ltm_state: torch.Tensor
    kv_cache: KVCache
    layer_idx: int


class AttentionSubLayer(nn.Module):
    """
    Encapsulates the Multi-Head Attention sub-layer with normalization and optional FiLM.
    """

    def __init__(self, config: DecoderBlockConfig, linear_class: nn.Module) -> None:
        """
        Initializes the AttentionSubLayer.

        Args:
            config: Configuration for the decoder block.
            linear_class: The linear layer class to use (e.g., nn.Linear or Linear4bit).
        """
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mha = self._create_mha(config, linear_class)
        self.dropout = Dropout(config.dropout_rate)
        self.film = (
            FiLMLayer(config.d_model, linear_class=linear_class)
            if config.long_term_memory
            else None
        )

        # LayerScale parameter: initialized to a small value for deep models
        self.layer_scale = nn.Parameter(torch.ones(config.d_model) * 1e-5)

    def _create_mha(
        self, config: DecoderBlockConfig, linear_class: nn.Module
    ) -> MultiHeadAttention:
        """
        Creates the Multi-Head Attention module.

        Args:
            config: Configuration for the decoder block.
            linear_class: The linear layer class to use.

        Returns:
            An initialized MultiHeadAttention instance.
        """
        mha_config = MultiHeadAttentionConfig(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_emb=config.rotary_emb,
            bias=False,
            num_layers=config.num_layers,
        )
        return MultiHeadAttention(mha_config, linear_class=linear_class)

    def forward(self, inputs: AttentionSubLayerInput) -> torch.Tensor:
        """
        Forward pass for the attention sub-layer using the input dataclass.

        Args:
            inputs: AttentionSubLayerInput object.

        Returns:
            The output tensor after attention and residual connection.
        """
        return self.forward_direct(
            inputs.x, inputs.ltm_state, inputs.kv_cache, inputs.layer_idx
        )

    def forward_direct(
        self,
        x: torch.Tensor,
        ltm_state: torch.Tensor,
        kv_cache: KVCache | None,
        layer_idx: int | None,
        skip_norm: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass directly using parameters to avoid object creation overhead.

        Args:
            x: Input hidden states.
            ltm_state: Long-Term Memory state.
            kv_cache: Key-Value cache.
            layer_idx: Index of the layer.
            skip_norm: Whether to skip internal normalization and residual.

        Returns:
            The output tensor after attention (and optional residual).
        """
        if skip_norm:
            # Parallel path: skip internal norm/film and residual sum
            attn_output = self.mha(x, kv_cache=kv_cache, layer_idx=layer_idx)
            return self.layer_scale * self.dropout(attn_output)

        x_norm = self.norm(x)
        if self.film:
            x_norm = self.film(x_norm, ltm_state)
        attn_output = self.mha(x_norm, kv_cache=kv_cache, layer_idx=layer_idx)
        return x + self.layer_scale * self.dropout(attn_output)
