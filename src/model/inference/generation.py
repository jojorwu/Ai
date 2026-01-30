"""
Encapsulates generation-related utility methods for the Transformer model.
"""
import torch
from torch import nn
from src.utils.training import calculate_gradient_norm


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

        surprise = calculate_gradient_norm(long_term_memory.parameters())

        long_term_memory.zero_grad()
        return surprise
