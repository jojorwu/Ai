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
from src.model.layers.core.linear import Linear

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
        self.attention_sublayer = AttentionSubLayer(config, linear_class)
        self.ff_sublayer = FeedForwardSubLayer(config, self.use_moe, linear_class)

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
        Performs the forward pass directly to avoid object creation overhead.

        Args:
            x: Input hidden states.
            ltm_state: Long-Term Memory state.
            kv_cache: Key-Value cache.
            layer_idx: Index of the layer.
            dynamic_top_k: Number of experts to activate.

        Returns:
            A tuple of (output_tensor, auxiliary_loss).
        """
        x = self.attention_sublayer.forward_direct(x, ltm_state, kv_cache, layer_idx)
        x, aux_loss = self.ff_sublayer(x, ltm_state, dynamic_top_k)
        return x, aux_loss
