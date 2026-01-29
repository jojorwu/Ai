"""
Encapsulates generation-related utility methods for the Transformer model.
"""
import torch
from torch import nn


class GenerationMixin:
    """A mixin for generation-related utility methods."""

    def calculate_surprise(
        self, value: torch.Tensor, ltm_override: nn.Module | None = None
    ) -> float:
        """
        Calculates the 'surprise' metric for the LTM update mechanism.
        """
        long_term_memory = ltm_override or self.layers.long_term_memory
        if not long_term_memory or not value.requires_grad:
            return 0.0

        long_term_memory.zero_grad()
        value.backward(retain_graph=False)
        grad_tensors = [
            p.grad.detach()
            for p in long_term_memory.parameters()
            if p.grad is not None
        ]
        if not grad_tensors:
            return 0.0

        # Optimized norm calculation to avoid large temporary tensor concatenation.
        # ||[a, b]|| = sqrt(||a||^2 + ||b||^2)
        total_norm_sq = sum(t.pow(2).sum() for t in grad_tensors)
        surprise = torch.sqrt(total_norm_sq).item()
        long_term_memory.zero_grad()
        return surprise
