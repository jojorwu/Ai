"""
PyTorch implementation of the Feed-Forward Sub-Layer for a Transformer Decoder Block.
"""
from __future__ import annotations
from typing import Tuple

import torch
from torch import nn

from src.config.model_config import DecoderBlockConfig, FeedForwardConfig, MoEConfig
from src.model.layers.core.dropout import Dropout
from src.model.layers.moe.feed_forward import FeedForward
from src.model.layers.moe.moe import MixtureOfExperts
from src.model.layers.core.rms_norm import RMSNorm
from src.model.layers.core.film import FiLMLayer


class FeedForwardSubLayer(nn.Module):
    """
    Encapsulates the Feed-Forward Network sub-layer with optional MoE and FiLM.
    """

    def __init__(
        self, config: DecoderBlockConfig, use_moe: bool, linear_class: nn.Module
    ) -> None:
        """
        Initializes the FeedForwardSubLayer.

        Args:
            config: Configuration for the decoder block.
            use_moe: Whether to use Mixture of Experts instead of standard FFN.
            linear_class: The linear layer class to use.
        """
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.ff_layer = self._create_ff_layer(config, use_moe, linear_class)
        self.dropout = Dropout(config.dropout_rate)
        self.film = (
            FiLMLayer(config.d_model, linear_class=linear_class)
            if config.long_term_memory
            else None
        )
        self.use_moe = use_moe

        # LayerScale parameter: initialized to a small value for deep models
        self.layer_scale = nn.Parameter(torch.ones(config.d_model) * 1e-5)

    def _create_ff_layer(
        self, config: DecoderBlockConfig, use_moe: bool, linear_class: nn.Module
    ) -> nn.Module:
        """
        Creates the feed-forward or MoE layer.

        Args:
            config: Configuration for the decoder block.
            use_moe: Whether to use MoE.
            linear_class: The linear layer class to use.

        Returns:
            The initialized feed-forward or MoE module.
        """
        if use_moe:
            moe_config = MoEConfig(
                d_model=config.d_model,
                d_ff=config.d_ff,
                num_experts=config.num_experts,
                top_k=config.top_k_experts,
                bias=False,
                use_shared_expert=getattr(config, "use_shared_expert", False),
            )
            return MixtureOfExperts(moe_config, linear_class=linear_class)
        ffn_config = FeedForwardConfig(
            d_model=config.d_model, d_ff=config.d_ff, bias=False, num_layers=config.num_layers
        )
        return FeedForward(ffn_config, linear_class=linear_class)

    def forward(
        self, x: torch.Tensor, ltm_state: torch.Tensor, dynamic_top_k: int | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the feed-forward sub-layer.

        Args:
            x: Input hidden states.
            ltm_state: Long-Term Memory state for FiLM.
            dynamic_top_k: Optional override for MoE experts.

        Returns:
            A tuple of (output_tensor, auxiliary_loss).
        """
        return self.forward_direct(x, ltm_state, dynamic_top_k)

    def forward_direct(
        self,
        x: torch.Tensor,
        ltm_state: torch.Tensor,
        dynamic_top_k: int | None,
        skip_norm: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass directly using parameters to avoid object creation overhead.

        Args:
            x: Input hidden states.
            ltm_state: Long-Term Memory state.
            dynamic_top_k: Number of experts to activate.
            skip_norm: Whether to skip internal normalization and residual.

        Returns:
            A tuple of (output_tensor, auxiliary_loss).
        """
        aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        if skip_norm:
            # Parallel path: skip internal norm/film and residual sum
            if self.use_moe:
                ffn_output, aux_loss = self.ff_layer(
                    x, dynamic_top_k=dynamic_top_k, ltm_context=ltm_state
                )
            else:
                ffn_output = self.ff_layer(x)
            return self.layer_scale * self.dropout(ffn_output), aux_loss

        x_norm = self.norm(x)
        if self.film:
            x_norm = self.film(x_norm, ltm_state)
        if self.use_moe:
            ffn_output, aux_loss = self.ff_layer(
                x_norm, dynamic_top_k=dynamic_top_k, ltm_context=ltm_state
            )
        else:
            ffn_output = self.ff_layer(x_norm)
        return x + self.layer_scale * self.dropout(ffn_output), aux_loss
