"""
PyTorch implementation of the Mixture of Experts (MoE) layer.
"""
import torch
from torch import nn
from torch.nn import functional as F

from config import FeedForwardConfig, MoEConfig
from nn_components.feed_forward import FeedForward
from nn_components.linear import Linear


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) layer, migrated to PyTorch.
    """
    def __init__(self, config: MoEConfig, linear_class=Linear):
        super().__init__()
        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.gate = linear_class(config.d_model, config.num_experts, bias=config.bias)
        ffn_config = FeedForwardConfig(
            d_model=config.d_model,
            d_ff=config.d_ff,
            bias=config.bias
        )
        self.experts = nn.ModuleList(
            [FeedForward(ffn_config, linear_class=linear_class) for _ in range(config.num_experts)]
        )

    def forward(self, x: torch.Tensor, dynamic_top_k: int = None):
        """
        Forward pass through the MoE layer using vectorized operations.
        """
        batch_size, seq_len, d_model = x.shape
        x_reshaped = x.view(-1, d_model)

        router_logits = self.gate(x_reshaped)

        current_top_k = dynamic_top_k if dynamic_top_k is not None else self.top_k

        # Get top-k experts and their weights
        top_k_weights, top_k_indices = torch.topk(router_logits, current_top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1, dtype=torch.float32).to(x.dtype)

        # Compute auxiliary load balancing loss
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        p_i = router_probs.mean(dim=0)

        top_k_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
        f_i = top_k_mask.sum(dim=0).sum(dim=0) / (batch_size * seq_len)

        aux_loss = self.num_experts * (p_i * f_i).sum()

        # Vectorized routing
        final_output = torch.zeros_like(x_reshaped)

        # Flatten the top_k indices and weights
        flat_top_k_indices = top_k_indices.view(-1)
        flat_top_k_weights = top_k_weights.view(-1)

        # Expand x_reshaped to match the flattened top_k dimension
        expanded_x = x_reshaped.unsqueeze(1).expand(-1, current_top_k, -1).reshape(-1, d_model)

        # Use torch.zeros and scatter_add_ for an efficient gather-scatter operation
        expert_outputs = torch.zeros_like(expanded_x)
        for i, expert in enumerate(self.experts):
            expert_mask = (flat_top_k_indices == i)
            if expert_mask.any():
                expert_inputs = expanded_x[expert_mask]
                expert_outputs.masked_scatter_(expert_mask.unsqueeze(-1), expert(expert_inputs))

        # Weight the expert outputs
        weighted_outputs = expert_outputs * flat_top_k_weights.unsqueeze(-1)

        # Sum the outputs for each token
        final_output = weighted_outputs.view(-1, current_top_k, d_model).sum(dim=1)

        return final_output.view(batch_size, seq_len, d_model), aux_loss
