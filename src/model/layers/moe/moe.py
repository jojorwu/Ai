"""
PyTorch implementation of a high-performance Mixture of Experts (MoE) layer.
"""
from __future__ import annotations
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from src.config.model_config import FeedForwardConfig, MoEConfig
from src.model.layers.moe.feed_forward import FeedForward
from src.model.layers.core.linear import Linear


class MixtureOfExperts(nn.Module):
    """
    Implements a Sparsely-Gated Mixture of Experts layer with an optional Shared Expert.

    This implementation uses a router (gate) to assign tokens to a subset of
    available experts, maximizing model capacity while maintaining constant
    computational cost per token. Each expert is a SwiGLU FeedForward network.
    An optional Shared Expert can be used to capture general, non-specialized
    knowledge by processing all tokens.
    """

    def __init__(self, config: MoEConfig, linear_class: nn.Module = Linear) -> None:
        """
        Initializes the MixtureOfExperts layer.

        Args:
            config: Configuration for the MoE layer.
            linear_class: The linear layer class to use for experts.
        """
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.d_model = config.d_model

        # The gating network (router)
        self.gate = linear_class(config.d_model, config.num_experts, bias=config.bias)

        # The experts are implemented using the FeedForward (SwiGLU) class.
        ffn_config = FeedForwardConfig(
            d_model=config.d_model,
            d_ff=config.d_ff,
            bias=config.bias,
        )
        self.experts = nn.ModuleList(
            [
                FeedForward(ffn_config, linear_class=linear_class)
                for _ in range(config.num_experts)
            ]
        )

        # Optional shared expert that processes all tokens
        self.shared_expert = (
            FeedForward(ffn_config, linear_class=linear_class)
            if getattr(config, "use_shared_expert", False)
            else None
        )

    def _compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        top_k_indices: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Computes the auxiliary load balancing loss and Router Z-loss.

        Args:
            router_logits: Raw logits from the gate.
            top_k_indices: Indices of selected experts.
            batch_size: Batch size of the input.
            seq_len: Sequence length of the input.

        Returns:
            The combined auxiliary loss scalar.
        """
        # 1. Load balancing loss (Switch Transformer style)
        gate_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        p_i = gate_probs.mean(dim=0)
        top_k_mask = F.one_hot(
            top_k_indices, num_classes=self.num_experts
        ).float()  # pylint: disable=not-callable
        f_i = top_k_mask.sum(dim=0).sum(dim=0) / (batch_size * seq_len)
        load_balancing_loss = self.num_experts * (p_i * f_i).sum()

        # 2. Router Z-loss (PaLM style)
        # Helps keep logits small and prevents overflow/instability
        # Loss = 0.001 * mean(logsumexp(logits)^2)
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()

        return load_balancing_loss + 0.001 * z_loss

    def forward(
        self, x: torch.Tensor, dynamic_top_k: int | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the MoE layer.

        Uses fully vectorized operations and groups tokens by expert for
        performance on parallel hardware.

        Args:
            x: Input tensor of shape [batch, seq, d_model].
            dynamic_top_k: Optional override for top-k selection.

        Returns:
            A tuple of (output_tensor, auxiliary_loss).
        """
        # pylint: disable=too-many-locals
        batch_size, seq_len, d_model = x.shape
        x_reshaped = x.view(-1, d_model)
        num_tokens = x_reshaped.shape[0]
        current_top_k = dynamic_top_k if dynamic_top_k is not None else self.top_k

        # 1. Route tokens to experts
        router_logits = self.gate(x_reshaped)

        if self.training:
            # Add small noise to router logits to encourage expert exploration
            noise = torch.randn_like(router_logits) * 0.01
            router_logits = router_logits + noise

        if current_top_k == 1:
            # Optimized path for top_k=1. We use the actual gate probability
            # to maintain differentiability.
            gate_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            routing_weights, selected_experts = gate_probs.max(dim=-1, keepdim=True)
        else:
            routing_weights, selected_experts = torch.topk(
                router_logits, current_top_k, dim=-1
            )
            routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)

        # 2. Compute auxiliary load balancing loss
        aux_loss = self._compute_aux_loss(
            router_logits, selected_experts, batch_size, seq_len
        )

        # 3. Create a flat tensor of expert outputs
        expert_outputs = torch.zeros(
            num_tokens * current_top_k, d_model, device=x.device, dtype=x.dtype
        )

        # 4. Use torch.gather to select inputs for each expert
        flat_selected_experts = selected_experts.view(-1)
        token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(
            current_top_k
        )

        # 5. Process tokens by each expert in batches
        expert_counts = torch.bincount(flat_selected_experts, minlength=self.num_experts)
        active_expert_indices = expert_counts.nonzero().flatten()
        active_expert_ids = active_expert_indices.tolist()

        for expert_idx in active_expert_ids:
            expert = self.experts[expert_idx]
            indices = torch.where(flat_selected_experts == expert_idx)[0]
            expert_result = expert(x_reshaped[token_indices[indices]])
            expert_outputs[indices] = expert_result

        # 6. Weight and combine the expert outputs
        weighted_outputs = expert_outputs * routing_weights.view(-1, 1)

        # 7. Use torch.scatter_add_ to sum the outputs for each token
        final_output = torch.zeros_like(x_reshaped)
        final_output.scatter_add_(
            0, token_indices.unsqueeze(-1).expand(-1, d_model), weighted_outputs
        )

        # 8. Add shared expert output if enabled
        if self.shared_expert is not None:
            final_output = final_output + self.shared_expert(x_reshaped)

        return final_output.view(batch_size, seq_len, d_model), aux_loss
