"""
Implements modular logit sampling techniques for the Transformer model.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F


class LogitSampler:
    """
    Handles sampling tokens from logits using a pipeline of transformations.

    This class provides modular methods for temperature scaling, top-k,
    top-p (nucleus), min-p filtering, and soft-clamping.
    """

    @staticmethod
    def apply_soft_cap(logits: torch.Tensor, soft_cap: float) -> torch.Tensor:
        """Applies Tanh-based soft-clamping to logits."""
        return soft_cap * torch.tanh(logits / soft_cap)

    @staticmethod
    def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Applies temperature scaling to logits."""
        if temperature == 1.0:
            return logits
        return logits / temperature

    @staticmethod
    def apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
        """Applies Min-P filtering to logits."""
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        logits = logits.clone()
        logits[probs < min_p * max_prob] = -float("Inf")
        return logits

    @staticmethod
    def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Applies Top-K filtering to logits."""
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.clone()
        logits[logits < v[..., -1, None]] = -float("Inf")
        return logits

    @staticmethod
    def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Applies Top-P (nucleus) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter the mask back to the original logits
        indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.clone()
        logits[indices_to_remove] = -float("Inf")
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        min_p: float = 0.0,
        logit_soft_cap: float | None = None,
    ) -> torch.Tensor:
        """
        Samples a token from the given logits using a transformation pipeline.

        Args:
            logits: Raw logits [batch, vocab].
            temperature: Randomness control. 0.0 for greedy sampling.
            top_k: Keep only top-k most likely tokens.
            top_p: Nucleus sampling threshold.
            min_p: Min-P sampling threshold.
            logit_soft_cap: Threshold for logit soft-clamping.

        Returns:
            A tensor containing the index of the sampled token [batch, 1].
        """
        # 1. Soft-clamping
        if logit_soft_cap is not None:
            logits = self.apply_soft_cap(logits, logit_soft_cap)

        # 2. Greedy shortcut
        if temperature == 0.0:
            _, next_token = torch.topk(logits, k=1, dim=-1)
            return next_token

        # 3. Temperature scaling
        logits = self.apply_temperature(logits, temperature)

        # 4. Min-P filtering
        if min_p > 0.0:
            logits = self.apply_min_p(logits, min_p)

        # 5. Top-K filtering
        if top_k > 0:
            logits = self.apply_top_k(logits, top_k)

        # 6. Top-P filtering
        if top_p > 0.0:
            logits = self.apply_top_p(logits, top_p)

        # 7. Final Sampling
        probs = F.softmax(logits, dim=-1)

        # Safety check: if all probabilities are zero or NaN fallback to greedy
        if torch.isnan(probs).any() or (probs.sum(dim=-1) <= 0).any():
            _, next_token = torch.topk(logits, k=1, dim=-1)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        return next_token
