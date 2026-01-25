"""
PyTorch implementation of a single Transformer Decoder Block.
"""
from dataclasses import dataclass

import torch
from bitsandbytes.nn import Linear4bit
from torch import nn

from src.config.model_config import DecoderBlockConfig
from src.model.layers.attention_sublayer import (AttentionSubLayer,
                                                 AttentionSubLayerInput)
from src.model.layers.ff_sublayer import FeedForwardSubLayer
from src.model.layers.linear import Linear


@dataclass
class ForwardPassInput:
    """Dataclass for storing inputs to the forward pass of a DecoderBlock."""
    x: torch.Tensor
    ltm_state: torch.Tensor
    kv_cache: 'KVCache' = None
    images: torch.Tensor = None
    layer_idx: int = None
    dynamic_top_k: int = None


class DecoderBlock(nn.Module):
    """
    Implements a single Transformer Decoder block, migrated to PyTorch.
    """
    def __init__(self, config: DecoderBlockConfig):
        super().__init__()
        self.use_moe = (
            config.num_experts is not None and config.top_k_experts is not None and
            config.num_experts > 0
        )
        linear_class = Linear4bit if config.load_in_4bit else Linear
        self.attention_sublayer = AttentionSubLayer(config, linear_class)
        self.ff_sublayer = FeedForwardSubLayer(config, self.use_moe, linear_class)

    def forward(self, inputs: ForwardPassInput):
        """Performs the forward pass of the Decoder Block."""
        attn_inputs = AttentionSubLayerInput(
            x=inputs.x,
            ltm_state=inputs.ltm_state,
            kv_cache=inputs.kv_cache,
            layer_idx=inputs.layer_idx,
        )
        x = self.attention_sublayer(attn_inputs)
        x, aux_loss = self.ff_sublayer(
            x, inputs.ltm_state, inputs.dynamic_top_k
        )
        return x, aux_loss
