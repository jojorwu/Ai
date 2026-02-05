"""
PyTorch implementation of the Long-Term Memory (LTM) module using
Multi-Head Gated Linear Associative Memory with learnable decay.
"""
from __future__ import annotations
import torch
from torch import nn

from src.model.layers.core.linear import Linear
from src.model.layers.core.rms_norm import RMSNorm


class LongTermMemory(nn.Module):
    """
    Multi-Head Gated Linear Associative Memory (Fast Weights) implementation for LTM.

    This architecture uses input-dependent gating to manage information
    persistence and retrieval from an associative memory matrix, with multiple
    heads for enhanced representational capacity and a learnable decay bias.
    """

    def __init__(
        self, d_model: int, d_hidden: int, num_layers: int = 1, num_heads: int = 4
    ) -> None:
        """
        Initializes the LongTermMemory module.

        Args:
            d_model: Dimension of the model's hidden states.
            d_hidden: Total dimension of the associative memory space.
            num_layers: Number of layers (not used in current simplified gated version).
            num_heads: Number of retrieval heads.
        """
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.head_dim = d_hidden // num_heads

        if d_hidden % num_heads != 0:
            raise ValueError("d_hidden must be divisible by num_heads.")

        # GLU-based Projections to associative space
        # We project to 2x the dimension to support gating (split into gate and value)
        self.q_proj = Linear(d_model, 2 * d_hidden, bias=False)
        self.k_proj = Linear(d_model, 2 * d_hidden, bias=False)
        self.v_proj = Linear(d_model, 2 * d_model, bias=False)

        # Gating projections for selective forgetting and updating
        # We separate the linear layer to inject a learnable decay bias
        self.forget_gate_proj = Linear(d_model, d_hidden)
        self.input_gate = nn.Sequential(Linear(d_model, d_hidden), nn.Sigmoid())

        # Learnable decay bias per head to provide a baseline forgetting rate
        # Initialized to a small positive value so that sigmoid starts around 0.5-0.9
        self.decay_bias = nn.Parameter(torch.ones(1, num_heads, 1, self.head_dim) * 2.0)

        # Retrieval normalization for stability
        self.retrieval_norm = RMSNorm(d_model)

        # Output projection to merge heads
        self.out_proj = Linear(d_model, d_model, bias=False)

        # Complexity head processes the query to determine sequence difficulty
        self.complexity_head = nn.Sequential(
            Linear(d_hidden, d_hidden // 2),
            nn.Tanh(),
            Linear(d_hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, prev_mem: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Multi-Head Gated Linear Associative LTM.

        Args:
            x: Input tensor summary of shape [batch, 1, d_model].
            prev_mem: Previous memory matrix of shape [batch, num_heads, head_dim, v_head_dim].

        Returns:
            A tuple containing:
                - context: Retrieved information [batch, 1, d_model].
                - complexity_score: Difficulty estimate [batch, 1, 1].
                - new_mem: Updated memory matrix [batch, num_heads, head_dim, v_head_dim].

        Raises:
            ValueError: If the input tensor x does not have the expected shape.
        """
        if x.ndim != 3 or x.size(1) != 1:
            raise ValueError(
                f"Expected input x of shape [batch, 1, d_model], got {list(x.shape)}"
            )

        batch_size = x.size(0)

        # 1. GLU-based Projections: SiLU(gate) * value
        q_gate, q_val = self.q_proj(x).chunk(2, dim=-1)
        q = torch.nn.functional.silu(q_gate) * q_val  # [batch, 1, d_hidden]

        k_gate, k_val = self.k_proj(x).chunk(2, dim=-1)
        k = torch.nn.functional.silu(k_gate) * k_val  # [batch, 1, d_hidden]

        v_gate, v_val = self.v_proj(x).chunk(2, dim=-1)
        v = torch.nn.functional.silu(v_gate) * v_val  # [batch, 1, d_model]

        # 2. Reshape for Multi-Head: [batch, heads, 1, head_dim]
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        v_heads = v.view(batch_size, 1, self.num_heads, -1).transpose(1, 2)
        v_head_dim = self.d_model // self.num_heads

        # Compute gates [batch, heads, 1, head_dim]
        # Forget gate incorporates learnable decay bias
        f_proj = self.forget_gate_proj(x).view(
            batch_size, 1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        f = torch.sigmoid(f_proj + self.decay_bias)

        i = self.input_gate(x).view(
            batch_size, 1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        if prev_mem is None:
            prev_mem = torch.zeros(
                batch_size,
                self.num_heads,
                self.head_dim,
                v_head_dim,
                device=x.device,
                dtype=x.dtype,
            )

        # 3. Gated update: M_t = f * M_{t-1} + i * (k^T @ v)
        kv_prod = torch.matmul(k.transpose(-2, -1), v_heads)
        new_mem = f.transpose(-2, -1) * prev_mem + i.transpose(-2, -1) * kv_prod

        # Numerical stability
        mem_norm = new_mem.norm(dim=(-2, -1), keepdim=True)
        new_mem = new_mem / (mem_norm.clamp(min=1.0) + 1e-6)
        new_mem = torch.clamp(new_mem, -1e4, 1e4)

        # 4. Retrieval: context = q @ M_t
        context = torch.matmul(q, new_mem)

        # 5. Merge heads: [batch, 1, d_model]
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, 1, self.num_heads * v_head_dim)
        )

        # Final projection and norm
        context = self.retrieval_norm(context)
        context = self.out_proj(context)

        # Complexity head uses the full query to score the input
        complexity_score = self.complexity_head(
            q.transpose(1, 2).reshape(batch_size, 1, -1)
        )

        return context, complexity_score, new_mem
