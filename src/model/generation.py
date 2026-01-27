"""
A mixin class for the Transformer model that encapsulates generation-related methods.
"""
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple


class GenerationMixin:
    """A mixin for generation-related methods."""

    def _sample_from_logits(self, logits, temperature, top_k, top_p):
        """Samples a token from logits using temperature, top-k, and top-p."""
        if temperature == 0.0:
            _, next_token = torch.topk(logits, k=1, dim=-1)
            return next_token

        logits = logits / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1, None]] = -float('Inf')

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token

    def _calculate_surprise(
        self, value: torch.Tensor, ltm_override: nn.Module | None = None
    ) -> float:
        """
        Calculates the 'surprise' metric for the LTM update mechanism.

        "Surprise" is a heuristic used to decide when to update the Long-Term
        Memory.
        It's defined as the norm of the gradients of the LTM's parameters with respect
        to the value head's output. A high surprise value indicates that a small change
        in the LTM would have a large impact on the predicted value, suggesting that the
        LTM's current state is "surprising" or ill-suited for the current context.
        This triggers an LTM update during generation.

        Args:
            value: The scalar tensor output from the value head. It must require gradients.
            ltm_override: The specific LTM module to use (for agent-based training).

        Returns:
            A float representing the calculated surprise value (gradient norm).
        """
        long_term_memory = ltm_override or self.layers.long_term_memory
        if not long_term_memory or not value.requires_grad:
            return 0.0

        long_term_memory.zero_grad()
        value.backward(retain_graph=True)
        grad_tensors = [
            p.grad.detach()
            for p in long_term_memory.parameters()
            if p.grad is not None
        ]
        if not grad_tensors:
            return 0.0

        surprise = torch.linalg.norm(
            torch.cat([t.flatten() for t in grad_tensors])
        ).item()
        long_term_memory.zero_grad()
        return surprise

    def _generate_speculative_chunk(
        self, draft_model: "Transformer", tokens: torch.Tensor, inputs: "GenerateInput"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates a small 'draft' chunk of tokens using a faster, smaller model."""
        draft_tokens = tokens
        with torch.no_grad():
            for _ in range(inputs.speculative_config.speculative_steps):
                draft_logits, _, _ = draft_model(
                    draft_tokens[:, -self.config.model.max_seq_len :]
                )
                next_token = self._sample_from_logits(
                    draft_logits[:, -1, :],
                    inputs.sampling_config.temperature,
                    inputs.sampling_config.dynamic_top_k or inputs.sampling_config.top_k,
                    inputs.sampling_config.top_p,
                )
                draft_tokens = torch.cat((draft_tokens, next_token), dim=1)
        return draft_tokens[:, tokens.size(1) :], draft_tokens

    def _validate_and_accept_chunk(
        self,
        true_logits: torch.Tensor,
        speculative_chunk: torch.Tensor,
        inputs: "GenerateInput",
    ) -> torch.Tensor | None:
        """Validates a speculative chunk against logits from the main model."""
        if speculative_chunk.size(0) != 1:
            raise NotImplementedError(
                "Speculative decoding only supports batch size 1."
            )

        verification_tokens = self._sample_from_logits(
            true_logits.view(-1, true_logits.size(-1)),
            inputs.sampling_config.temperature,
            inputs.sampling_config.dynamic_top_k or inputs.sampling_config.top_k,
            inputs.sampling_config.top_p,
        ).view(speculative_chunk.shape)

        mismatches = (speculative_chunk != verification_tokens).long()

        if not mismatches.any():
            return speculative_chunk

        first_mismatch_idx = torch.argmax(mismatches, dim=1)[0]

        if first_mismatch_idx == 0 and mismatches[0, 0] == 1:
            return verification_tokens[:, :1]

        accepted_prefix = speculative_chunk[:, :first_mismatch_idx]
        corrected_token = verification_tokens[
            :, first_mismatch_idx : first_mismatch_idx + 1
        ]

        return torch.cat([accepted_prefix, corrected_token], dim=1)
