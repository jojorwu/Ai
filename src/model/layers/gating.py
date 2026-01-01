"""
PyTorch implementation of the Gating Network for dynamic architecture selection.
"""
from torch import nn

from src.model.layers.linear import Linear


class GatingNetwork(nn.Module):
    """
    A small MLP that determines the number of layers and experts to use.
    """

    def __init__(self, d_model: int, num_layers: int, num_experts: int | None):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts

        self.network = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.layer_head = Linear(d_model, num_layers)
        # The expert head is only created if the model uses MoE.
        self.expert_head = (
            Linear(d_model, num_experts) if num_experts is not None else None
        )

    def forward(self, x):
        """
        Forward pass for the Gating Network.
        """
        hidden = self.network(x)
        layer_logits = self.layer_head(hidden)
        active_layers = layer_logits.argmax(dim=-1) + 1

        # If there's no expert head (i.e., no MoE), return a default value.
        if self.expert_head is None:
            top_k_experts = 1  # Default value, will be ignored anyway.
        else:
            expert_logits = self.expert_head(hidden)
            top_k_experts = expert_logits.argmax(dim=-1) + 1

        return active_layers, top_k_experts
