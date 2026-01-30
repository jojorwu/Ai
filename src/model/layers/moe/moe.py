"""
PyTorch implementation of the Mixture of Experts (MoE) layer.
"""
import torch
from torch import nn
from torch.nn import functional as F

from src.config.model_config import FeedForwardConfig, MoEConfig
from src.model.layers.moe.feed_forward import FeedForward
from src.model.layers.core.linear import Linear


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

    def _compute_aux_loss(self, router_logits, top_k_indices, batch_size, seq_len):
        """Computes the auxiliary load balancing loss."""
        # pylint: disable=too-many-locals
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        p_i = router_probs.mean(dim=0)
        top_k_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).float()  # pylint: disable=not-callable
        f_i = top_k_mask.sum(dim=0).sum(dim=0) / (batch_size * seq_len)
        return self.num_experts * (p_i * f_i).sum()

    def forward(self, x: torch.Tensor, dynamic_top_k: int = None):
        """
        Forward pass through the MoE layer using fully vectorized operations.
        This implementation avoids Python loops for better performance on parallel hardware.
        """
        # pylint: disable=too-many-locals
        batch_size, seq_len, d_model = x.shape
        x_reshaped = x.view(-1, d_model)
        num_tokens = x_reshaped.shape[0]
        current_top_k = dynamic_top_k if dynamic_top_k is not None else self.top_k

        # 1. Route tokens to experts
        router_logits = self.gate(x_reshaped)

        if current_top_k == 1:
            # Optimized path for top_k=1. We use the actual gate probability
            # to maintain differentiability and allow the router to scale expert contribution.
            gate_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            routing_weights, selected_experts = gate_probs.max(dim=-1, keepdim=True)
        else:
            routing_weights, selected_experts = torch.topk(
                router_logits, current_top_k, dim=-1
            )
            routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)

        # 2. Compute auxiliary load balancing loss
        aux_loss = self._compute_aux_loss(router_logits, selected_experts, batch_size, seq_len)

        # 3. Create a flat tensor of expert outputs
        expert_outputs = torch.zeros(
            num_tokens * current_top_k, d_model, device=x.device, dtype=x.dtype
        )

        # 4. Use torch.gather to select inputs for each expert
        # This creates a flat tensor of all tokens that need to be processed.
        flat_selected_experts = selected_experts.view(-1)
        token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(
            current_top_k
        )

        # 5. Process tokens by each expert in batches
        # We group tokens for the same expert to maximize GPU utilization.
        # Use nonzero() to find indices of experts that have at least one token assigned.
        expert_counts = torch.bincount(flat_selected_experts, minlength=self.num_experts)
        active_expert_indices = expert_counts.nonzero().flatten()

        # We perform ONE synchronization to get the list of active expert IDs.
        # This is much faster than calling .item() inside the loop.
        active_expert_ids = active_expert_indices.tolist()

        for expert_idx in active_expert_ids:
            expert = self.experts[expert_idx]

            # Find which token-expert pairs in the flat list belong to this expert.
            indices = torch.where(flat_selected_experts == expert_idx)[0]

            # Run the expert on its batch of tokens.
            expert_result = expert(x_reshaped[token_indices[indices]])

            # Store results using direct indexing.
            expert_outputs[indices] = expert_result

        # 6. Weight and combine the expert outputs
        weighted_outputs = expert_outputs * routing_weights.view(-1, 1)

        # 7. Use torch.scatter_add_ to sum the outputs for each token
        final_output = torch.zeros_like(x_reshaped)
        final_output.scatter_add_(
            0, token_indices.unsqueeze(-1).expand(-1, d_model), weighted_outputs
        )

        return final_output.view(batch_size, seq_len, d_model), aux_loss
