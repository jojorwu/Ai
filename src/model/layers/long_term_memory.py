"""
PyTorch implementation of the Long-Term Memory (LTM) module using Linear Associative Memory.
"""
import torch
from torch import nn

from src.model.layers.linear import Linear


class LongTermMemory(nn.Module):
    """
    Linear Associative Memory (Fast Weights) implementation for LTM.
    This architecture uses key-value associations to store and retrieve information.
    """
    def __init__(self, d_model: int, d_hidden: int, num_layers: int = 1):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden

        # Projections to associative space
        self.q_proj = Linear(d_model, d_hidden, bias=False)
        self.k_proj = Linear(d_model, d_hidden, bias=False)
        self.v_proj = Linear(d_model, d_model, bias=False)

        # Learnable decay for the memory matrix
        self.memory_decay = nn.Parameter(torch.tensor(0.9))

        # Complexity head processes the query to determine sequence difficulty
        # Use a small MLP if d_hidden is large enough, otherwise simple linear
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
        Forward pass for the Linear Associative LTM.
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

        if prev_mem is None:
            prev_mem = torch.zeros(
                batch_size, self.d_hidden, self.d_model, device=x.device, dtype=x.dtype
            )

        # Update memory: M_t = decay * M_{t-1} + k^T @ v
        # This stores the association between the key and the value.
        new_mem = self.memory_decay * prev_mem + torch.matmul(k.transpose(1, 2), v)

        # Retrieval: context = q @ M_t
        context = torch.matmul(q, new_mem)  # [batch, 1, d_model]

        # Complexity head uses the query to score the input
        complexity_score = self.complexity_head(q)

        return context, complexity_score, new_mem
