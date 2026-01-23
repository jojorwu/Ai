"""
PyTorch implementation of a single Transformer Decoder Block.
"""
from dataclasses import dataclass

import torch
from torch import nn
from bitsandbytes.nn import Linear4bit

from src.config.model_config import (DecoderBlockConfig, FeedForwardConfig, MoEConfig,
                        MultiHeadAttentionConfig)
from src.model.layers.dropout import Dropout
from src.model.layers.feed_forward import FeedForward
from src.model.layers.linear import Linear
from src.model.layers.moe import MixtureOfExperts
from src.model.layers.multi_head_attention import MultiHeadAttention
from src.model.layers.rms_norm import RMSNorm
from src.model.layers.film import FiLMLayer


@dataclass
class ForwardPassInput:
    """Dataclass for storing inputs to the forward pass of a DecoderBlock."""
    x: torch.Tensor
    ltm_state: torch.Tensor
    kv_cache: 'KVCache' = None
    images: torch.Tensor = None
    layer_idx: int = None
    dynamic_top_k: int = None


@dataclass
class AttentionSubLayerInput:
    """Inputs for the AttentionSubLayer."""
    x: torch.Tensor
    ltm_state: torch.Tensor
    kv_cache: 'KVCache'
    layer_idx: int

class AttentionSubLayer(nn.Module):
    """Encapsulates the Multi-Head Attention sub-layer."""
    def __init__(self, config: DecoderBlockConfig, linear_class):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mha = self._create_mha(config, linear_class)
        self.dropout = Dropout(config.dropout_rate)
        self.film = (
            FiLMLayer(config.d_model, linear_class=linear_class)
            if config.long_term_memory else None
        )

    def _create_mha(self, config, linear_class):
        mha_config = MultiHeadAttentionConfig(
            d_model=config.d_model, num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads, rotary_emb=config.rotary_emb,
            bias=False, num_layers=config.num_layers
        )
        return MultiHeadAttention(mha_config, linear_class=linear_class)

    def forward(self, inputs: AttentionSubLayerInput):
        """Forward pass for the attention sub-layer."""
        x_norm = self.norm(inputs.x)
        if self.film:
            x_norm = self.film(x_norm, inputs.ltm_state)
        attn_output = self.mha(
            x_norm, kv_cache=inputs.kv_cache, layer_idx=inputs.layer_idx
        )
        return inputs.x + self.dropout(attn_output)

class FeedForwardSubLayer(nn.Module):
    """Encapsulates the Feed-Forward Network sub-layer."""
    def __init__(self, config: DecoderBlockConfig, use_moe: bool, linear_class):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.ff_layer = self._create_ff_layer(config, use_moe, linear_class)
        self.dropout = Dropout(config.dropout_rate)
        self.film = (
            FiLMLayer(config.d_model, linear_class=linear_class)
            if config.long_term_memory else None
        )
        self.use_moe = use_moe

    def _create_ff_layer(self, config, use_moe, linear_class):
        if use_moe:
            moe_config = MoEConfig(
                d_model=config.d_model, d_ff=config.d_ff,
                num_experts=config.num_experts, top_k=config.top_k_experts,
                bias=False
            )
            return MixtureOfExperts(moe_config, linear_class=linear_class)
        ffn_config = FeedForwardConfig(
            d_model=config.d_model, d_ff=config.d_ff, bias=False,
            num_layers=config.num_layers
        )
        return FeedForward(ffn_config, linear_class=linear_class)

    def forward(self, x, ltm_state, dynamic_top_k):
        """Forward pass for the feed-forward sub-layer."""
        aux_loss = torch.tensor(0.0, device=x.device)
        x_norm = self.norm(x)
        if self.film:
            x_norm = self.film(x_norm, ltm_state)
        if self.use_moe:
            ffn_output, aux_loss = self.ff_layer(
                x_norm, dynamic_top_k=dynamic_top_k
            )
        else:
            ffn_output = self.ff_layer(x_norm)
        return x + self.dropout(ffn_output), aux_loss


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
