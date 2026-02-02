"""
Implements logit sampling techniques for the Transformer model.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F


class LogitSampler:
    """
    Handles sampling tokens from logits.

    This class provides various methods such as temperature sampling,
    top-k, top-p (nucleus), and min-p filtering.
    """

    @staticmethod
    def sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        min_p: float = 0.0,
    ) -> torch.Tensor:
        """
        Samples a token from the given logits.

        Args:
            logits: Raw logits from the model [batch, seq, vocab] or [batch, vocab].
            temperature: Randomness control. 0.0 for greedy sampling.
            top_k: Keep only top-k most likely tokens.
            top_p: Keep only tokens with cumulative probability up to top_p.
            min_p: Keep tokens with probability at least min_p * max_prob.

        Returns:
            A tensor containing the index of the sampled token [batch, 1].
        """
        if temperature == 0.0:
            # Greedy sampling
            _, next_token = torch.topk(logits, k=1, dim=-1)
            return next_token

        # Apply temperature
        logits = logits / temperature

        # Apply Min-P filtering
        if min_p > 0.0:
            probs = F.softmax(logits, dim=-1)
            max_prob = probs.max(dim=-1, keepdim=True).values
            logits[probs < min_p * max_prob] = -float("Inf")

        # Apply top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # Any logit smaller than the last of the top-k is set to -inf
            logits[logits < v[..., -1, None]] = -float("Inf")

        # Apply top-p (nucleus) filtering
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter the mask back to the original logits.
            # We use a zero tensor as the base to ensure a clean mask.
            indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float("Inf")

        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)

        # Safety check: if all probabilities are zero or NaN
        # fallback to greedy sampling from the original (unfiltered) logits.
        if torch.isnan(probs).any() or (probs.sum(dim=-1) <= 0).any():
            _, next_token = torch.topk(logits, k=1, dim=-1)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        return next_token
