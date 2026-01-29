"""
PyTorch implementation of the Gating Network for dynamic architecture selection.
"""
import torch
from torch import nn

from src.model.layers.core.linear import Linear


class GatingNetwork(nn.Module):
    """
    A small MLP that determines the number of layers and experts to use.
    Uses LayerNorm and GELU for improved stability and non-linearity.
    """

    def __init__(self, d_model: int, num_layers: int, num_experts: int | None):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts

        self.network = nn.Sequential(
            Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.layer_head = Linear(d_model, num_layers)
        # The expert head is only created if the model uses MoE.
        self.expert_head = (
            Linear(d_model, num_experts) if num_experts is not None else None
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the Gating Network.
        Args:
            x: Input tensor, typically from LTM state [batch, 1, d_model]
        Returns:
            active_layers: Number of decoder layers to use.
            top_k_experts: Number of experts to activate per layer.
        """
        hidden = self.network(x)

        # Determine number of layers (1 to num_layers)
        layer_logits = self.layer_head(hidden)
        active_layers = layer_logits.argmax(dim=-1) + 1

        # Determine number of experts (1 to num_experts)
        if self.expert_head is None:
            top_k_experts = 1
        else:
            expert_logits = self.expert_head(hidden)
            top_k_experts = expert_logits.argmax(dim=-1) + 1

        return active_layers, top_k_experts
