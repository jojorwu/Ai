"""
PyTorch implementation of the Gating Network for dynamic architecture selection.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

from src.model.layers.core.linear import Linear
from src.model.layers.core.rms_norm import RMSNorm


class GatingNetwork(nn.Module):
    """
    A small MLP that determines the number of layers and experts to use.

    Uses RMSNorm and GELU for improved stability. It takes the retrieved
    LTM context and an optional complexity score to predict dynamic
    architectural parameters.
    """

    def __init__(self, d_model: int, num_layers: int, num_experts: int | None) -> None:
        """
        Initializes the GatingNetwork.

        Args:
            d_model: Dimension of the input hidden states.
            num_layers: Maximum number of decoder layers available.
            num_experts: Maximum number of experts available (if using MoE).
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts

        self.network = nn.Sequential(
            Linear(d_model, d_model),
            RMSNorm(d_model),
            nn.GELU(),
            Linear(d_model, d_model),
            RMSNorm(d_model),
        )

        # Head to predict the number of active layers
        self.layer_head = Linear(d_model, num_layers)

        # Head to predict the number of experts to activate
        self.expert_head = (
            Linear(d_model, num_experts) if num_experts is not None else None
        )

        # Project complexity score to d_model if provided
        self.complexity_proj = Linear(1, d_model)

    def forward(
        self, x: torch.Tensor, complexity_score: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Gating Network.

        Args:
            x: Input tensor from LTM state [batch, 1, d_model].
            complexity_score: Optional difficulty estimate from LTM [batch, 1, 1].

        Returns:
            A tuple of (active_layers, top_k_experts) as tensors.
            These tensors maintain differentiability via a straight-through proxy.
        """
        # Incorporate complexity score if provided
        if complexity_score is not None:
            # Reshape complexity score to match x's batch and seq dimensions
            comp_feat = self.complexity_proj(complexity_score)
            x = x + comp_feat

        hidden = self.network(x)

        # 1. Determine number of layers
        layer_logits = self.layer_head(hidden)
        active_layers = self._differentiable_selection(layer_logits, self.num_layers)

        # 2. Determine number of experts
        if self.expert_head is None:
            top_k_experts = torch.ones_like(active_layers)
        else:
            expert_logits = self.expert_head(hidden)
            top_k_experts = self._differentiable_selection(
                expert_logits, self.num_experts
            )

        return active_layers, top_k_experts

    def _differentiable_selection(self, logits: torch.Tensor, max_val: int) -> torch.Tensor:
        """
        Performs discrete selection while maintaining a differentiable path.

        Uses a weighted sum as a continuous proxy and combines it with
        the discrete argmax using the straight-through estimator trick.

        Args:
            logits: Logits for the selection head [batch, 1, max_val].
            max_val: Maximum value to select.

        Returns:
            A tensor containing the selected values (1 to max_val).
        """
        # Discrete selection
        discrete = (logits.argmax(dim=-1) + 1).float()

        # Continuous proxy (weighted average)
        probs = F.softmax(logits, dim=-1)
        values = torch.arange(1, max_val + 1, device=logits.device, dtype=logits.dtype)
        continuous = (probs * values).sum(dim=-1)

        # Straight-through estimator: discrete value with continuous gradient
        return discrete + (continuous - continuous.detach())
