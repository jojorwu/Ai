"""
PyTorch implementation of a single Transformer Decoder Block.
"""
from dataclasses import dataclass

import torch
from torch import nn
from bitsandbytes.nn import Linear4bit

from config import (DecoderBlockConfig, FeedForwardConfig, MoEConfig,
                    MultiHeadAttentionConfig)
from nn_components.dropout import Dropout
from nn_components.feed_forward import FeedForward
from nn_components.linear import Linear
from nn_components.moe import MixtureOfExperts
from nn_components.multi_head_attention import MultiHeadAttention
from nn_components.rms_norm import RMSNorm


@dataclass
class ForwardPassInput:
    """Dataclass for storing inputs to the forward pass of a DecoderBlock."""
    x: torch.Tensor
    ltm_state: torch.Tensor
    mask: torch.Tensor = None
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
        self.config = config
        self.ltm = config.long_term_memory
        self.use_moe = self._check_moe_usage(config)

        linear_class = Linear4bit if config.load_in_4bit else Linear
        self.mha = self._create_mha(config, linear_class)
        self.ff_layer = self._create_ff_layer(config, linear_class)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.dropout = nn.ModuleDict(
            {'dropout1': Dropout(config.dropout_rate),
             'dropout2': Dropout(config.dropout_rate)}
        )

    def _check_moe_usage(self, config):
        return (config.num_experts is not None and
                config.top_k_experts is not None and
                config.num_experts > 0)

    def _create_mha(self, config, linear_class):
        mha_config = MultiHeadAttentionConfig(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_emb=config.rotary_emb,
            bias=False,
            num_layers=config.num_layers
        )
        return MultiHeadAttention(mha_config, linear_class=linear_class)

    def _create_ff_layer(self, config, linear_class):
        if self.use_moe:
            moe_config = MoEConfig(
                d_model=config.d_model,
                d_ff=config.d_ff,
                num_experts=config.num_experts,
                top_k=config.top_k_experts,
                bias=False
            )
            return MixtureOfExperts(moe_config, linear_class=linear_class)
        ffn_config = FeedForwardConfig(
            d_model=config.d_model,
            d_ff=config.d_ff,
            bias=False,
            num_layers=config.num_layers
        )
        return FeedForward(ffn_config, linear_class=linear_class)

    def forward(self, inputs: ForwardPassInput):
        """Performs the forward pass of the Decoder Block."""
        aux_loss = torch.tensor(0.0, device=inputs.x.device)

        # Additive memory injection before the first sub-layer
        x_with_mem = inputs.x + inputs.ltm_state if self.ltm else inputs.x
        x_norm1 = self.norm1(x_with_mem)

        attn_output = self.mha(
            x_norm1,
            mask=inputs.mask,
            kv_cache=inputs.kv_cache,
            layer_idx=inputs.layer_idx
        )

        # First residual connection
        x = inputs.x + self.dropout['dropout1'](attn_output)

        x_norm2 = self.norm2(x)

        if self.use_moe:
            ffn_output, aux_loss = self.ff_layer(
                x_norm2, dynamic_top_k=inputs.dynamic_top_k
            )
        else:
            ffn_output = self.ff_layer(x_norm2)

        # Second residual connection
        x = x + self.dropout['dropout2'](ffn_output)

        return x, aux_loss
