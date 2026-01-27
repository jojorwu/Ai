"""
PyTorch implementation of the Attention Sub-Layer for a Transformer Decoder Block.
"""
from dataclasses import dataclass
import torch
from torch import nn

from src.config.core import DecoderBlockConfig, MultiHeadAttentionConfig
from src.model.layers.dropout import Dropout
from src.model.layers.multi_head_attention import MultiHeadAttention
from src.model.layers.rms_norm import RMSNorm
from src.model.layers.film import FiLMLayer


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
