"""
PyTorch implementation of a single Transformer Decoder Block.
"""
import torch
import torch.nn as nn
from dataclasses import dataclass

from config import DecoderBlockConfig
from nn_components.dropout import Dropout
from nn_components.feed_forward import FeedForward
from nn_components.moe import MixtureOfExperts
from nn_components.multi_head_attention import MultiHeadAttention
from nn_components.rms_norm import RMSNorm
from nn_components.linear import Linear
from bitsandbytes.nn import Linear4bit


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


from config import DecoderBlockConfig, MultiHeadAttentionConfig, MoEConfig


class DecoderBlock(nn.Module):
    """
    Implements a single Transformer Decoder block, migrated to PyTorch.
    """
    def __init__(self, config: DecoderBlockConfig):
        super().__init__()
        self.config = config

        linear_class = Linear4bit if config.load_in_4bit else Linear

        mha_config = MultiHeadAttentionConfig(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_emb=config.rotary_emb,
            bias=False, # Typically no bias in MHA projections
            num_layers=config.num_layers
        )
        self.mha = MultiHeadAttention(mha_config, linear_class=linear_class)
        self.ltm = config.long_term_memory

        self.use_moe = (config.num_experts is not None and
                        config.top_k_experts is not None and
                        config.num_experts > 0)

        if self.use_moe:
            moe_config = MoEConfig(
                d_model=config.d_model,
                d_ff=config.d_ff,
                num_experts=config.num_experts,
                top_k=config.top_k_experts,
                bias=False # Typically no bias in MoE experts
            )
            self.moe_layer = MixtureOfExperts(moe_config, linear_class=linear_class)
        else:
            self.ffn = FeedForward(config.d_model, config.d_ff, bias=False, num_layers=config.num_layers, linear_class=linear_class)

        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.dropout1 = Dropout(config.dropout_rate)
        self.dropout2 = Dropout(config.dropout_rate)

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
        x = inputs.x + self.dropout1(attn_output)

        x_norm2 = self.norm2(x)

        if self.use_moe:
            ffn_output, aux_loss = self.moe_layer(x_norm2, dynamic_top_k=inputs.dynamic_top_k)
        else:
            ffn_output = self.ffn(x_norm2)

        # Second residual connection
        x = x + self.dropout2(ffn_output)

        return x, aux_loss
