"""
PyTorch implementation of a single Transformer Decoder Block.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch
from bitsandbytes.nn import Linear4bit
from torch import nn

from src.config.model_config import DecoderBlockConfig
from src.model.layers.attention.attention_sublayer import (
    AttentionSubLayer,
    AttentionSubLayerInput,
)
from src.model.layers.moe.ff_sublayer import FeedForwardSubLayer
from src.model.layers.attention.cross_attention import GatedCrossAttention
from src.model.layers.core.linear import Linear
from src.model.layers.core.rms_norm import RMSNorm
from src.model.layers.core.film import FiLMLayer

if TYPE_CHECKING:
    from src.model.layers.attention.kv_cache import KVCache


@dataclass
class ForwardPassInput:
    """
    Dataclass for storing inputs to the forward pass of a DecoderBlock.

    Attributes:
        x: Input tensor of shape [batch, seq_len, d_model].
        ltm_state: Long-Term Memory state tensor.
        kv_cache: Optional persistent Key-Value cache.
        images: Optional image embeddings.
        layer_idx: Index of the current layer.
        dynamic_top_k: Optional dynamic override for MoE experts.
    """
    x: torch.Tensor
    ltm_state: torch.Tensor
    kv_cache: KVCache | None = None
    images: torch.Tensor | None = None
    layer_idx: int | None = None
    dynamic_top_k: int | None = None


class DecoderBlock(nn.Module):
    """
    Implements a single Transformer Decoder block.

    Each block consists of a self-attention sub-layer followed by a
    feed-forward (or MoE) sub-layer, both with residual connections.
    """

    def __init__(self, config: DecoderBlockConfig) -> None:
        """
        Initializes the DecoderBlock.

        Args:
            config: Configuration object for the decoder block.
        """
        super().__init__()
        self.use_moe = (
            config.num_experts is not None
            and config.top_k_experts is not None
            and config.num_experts > 0
        )
        linear_class = Linear4bit if config.load_in_4bit else Linear

        # Shared normalization and optional FiLM for parallel architecture
        self.norm = RMSNorm(config.d_model)
        self.film = (
            FiLMLayer(config.d_model, linear_class=linear_class)
            if config.long_term_memory
            else None
        )

        self.attention_sublayer = AttentionSubLayer(config, linear_class)
        self.ff_sublayer = FeedForwardSubLayer(config, self.use_moe, linear_class)

        # Gated Cross-Attention for LTM Fusion
        self.ltm_cross_attn = (
            GatedCrossAttention(config.d_model, linear_class=linear_class)
            if config.long_term_memory
            else None
        )

        # Residual scaling for deep stability: learnable multiplier for all sub-layers
        # Initialized to 1.0 to start with standard behavior
        self.residual_scale = nn.Parameter(torch.ones(config.d_model))

    def forward(self, inputs: ForwardPassInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the Decoder Block.

        Args:
            inputs: ForwardPassInput object containing input tensors and metadata.

        Returns:
            A tuple of (output_tensor, auxiliary_loss).
        """
        return self.forward_direct(
            inputs.x,
            inputs.ltm_state,
            inputs.kv_cache,
            inputs.layer_idx,
            inputs.dynamic_top_k,
        )

    def forward_direct(
        self,
        x: torch.Tensor,
        ltm_state: torch.Tensor,
        kv_cache: KVCache | None,
        layer_idx: int | None,
        dynamic_top_k: int | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass directly using a parallel architecture.

        Args:
            x: Input hidden states.
            ltm_state: Long-Term Memory state.
            kv_cache: Key-Value cache.
            layer_idx: Index of the layer.
            dynamic_top_k: Number of experts to activate.

        Returns:
            A tuple of (output_tensor, auxiliary_loss).
        """
        # 1. Shared normalization and optional FiLM modulation
        x_norm = self.norm(x)
        if self.film:
            x_norm = self.film(x_norm, ltm_state)

        # 2. Parallel computation of Attention and FFN/MoE
        # Note: We pass pre-normalized input to sublayers
        attn_out = self.attention_sublayer.forward_direct(
            x_norm, ltm_state, kv_cache, layer_idx, skip_norm=True
        )
        ffn_out, aux_loss = self.ff_sublayer.forward_direct(
            x_norm, ltm_state, dynamic_top_k, skip_norm=True
        )

        # 3. Optional: Gated Cross-Attention to LTM context
        ltm_fused_out = torch.zeros_like(x)
        if self.ltm_cross_attn and ltm_state is not None:
            # Attend to LTM state
            ltm_fused_out = self.ltm_cross_attn(x_norm, ltm_state) - x_norm

        # 4. Combine with residual connection and learnable scaling
        # Note: individual LayerScale is still applied internally by sublayers
        combined_parallel_out = attn_out + ffn_out + ltm_fused_out
        x = x + self.residual_scale * combined_parallel_out

        return x, aux_loss
