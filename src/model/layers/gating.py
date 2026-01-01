"""
PyTorch implementation of the Gating Network for dynamic architecture selection.
"""
from torch import nn

from src.model.layers.linear import Linear


class GatingNetwork(nn.Module):
    """
    A small MLP that determines the number of layers and experts to use.
    """

    def __init__(self, d_model: int, num_layers: int, num_experts: int):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts

        self.network = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.layer_head = Linear(d_model, num_layers)
        self.expert_head = Linear(d_model, num_experts)

    def forward(self, x):
        """
        Forward pass for the Gating Network.
        """
        hidden = self.network(x)
        layer_logits = self.layer_head(hidden)
        expert_logits = self.expert_head(hidden)

        # Use argmax to select the number of layers and experts
        active_layers = layer_logits.argmax(dim=-1) + 1
        top_k_experts = expert_logits.argmax(dim=-1) + 1

        return active_layers, top_k_experts
