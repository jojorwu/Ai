"""
PyTorch implementation of the Long-Term Memory (LTM) module using Gated Linear Associative Memory.
"""
import torch
from torch import nn

from src.model.layers.core.linear import Linear
from src.model.layers.core.rms_norm import RMSNorm


class LongTermMemory(nn.Module):
    """
    Gated Linear Associative Memory (Fast Weights) implementation for LTM.
    This architecture uses input-dependent gating to manage information persistence.
    """
    def __init__(self, d_model: int, d_hidden: int, num_layers: int = 1):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden

        # Projections to associative space
        self.q_proj = Linear(d_model, d_hidden, bias=False)
        self.k_proj = Linear(d_model, d_hidden, bias=False)
        self.v_proj = Linear(d_model, d_model, bias=False)

        # Gating projections for selective forgetting and updating
        self.forget_gate = nn.Sequential(
            Linear(d_model, d_hidden),
            nn.Sigmoid()
        )
        self.input_gate = nn.Sequential(
            Linear(d_model, d_hidden),
            nn.Sigmoid()
        )

        # Retrieval normalization for stability
        self.retrieval_norm = RMSNorm(d_model)

        # Complexity head processes the query to determine sequence difficulty
        if d_hidden > 1:
            self.complexity_head = nn.Sequential(
                Linear(d_hidden, d_hidden // 2),
                nn.Tanh(),
                Linear(d_hidden // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.complexity_head = nn.Sequential(
                Linear(d_hidden, 1),
                nn.Sigmoid()
            )

    def forward(
        self, x: torch.Tensor, prev_mem: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Gated Linear Associative LTM.
        Args:
            x: Input tensor summary [batch, 1, d_model]
            prev_mem: Previous memory matrix [batch, d_hidden, d_model]
        Returns:
            context: Retrieved information [batch, 1, d_model]
            complexity_score: Difficulty estimate [batch, 1, 1]
            new_mem: Updated memory matrix [batch, d_hidden, d_model]
        """
        batch_size = x.size(0)

        q = self.q_proj(x)  # [batch, 1, d_hidden]
        k = self.k_proj(x)  # [batch, 1, d_hidden]
        v = self.v_proj(x)  # [batch, 1, d_model]

        # Compute gates [batch, 1, d_hidden]
        f = self.forget_gate(x)
        i = self.input_gate(x)

        if prev_mem is None:
            prev_mem = torch.zeros(
                batch_size, self.d_hidden, self.d_model, device=x.device, dtype=x.dtype
            )

        # Gated update: M_t = f * M_{t-1} + i * (k^T @ v)
        # We apply f row-wise to the memory matrix.
        new_mem = f.transpose(1, 2) * prev_mem + i.transpose(1, 2) * torch.matmul(k.transpose(1, 2), v)

        # Numerical stability: normalize the associative matrix to prevent weight explosion.
        # This keeps the values in the memory matrix within a reasonable range.
        mem_norm = new_mem.norm(dim=(1, 2), keepdim=True)
        new_mem = new_mem / (mem_norm.clamp(min=1.0) + 1e-6)
        new_mem = torch.clamp(new_mem, -1e4, 1e4)

        # Retrieval: context = q @ M_t
        context = torch.matmul(q, new_mem)  # [batch, 1, d_model]

        # Apply normalization to the retrieved context
        context = self.retrieval_norm(context)

        # Complexity head uses the query to score the input
        complexity_score = self.complexity_head(q)

        return context, complexity_score, new_mem
