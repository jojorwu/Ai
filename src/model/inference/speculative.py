"""
Implements speculative decoding logic for the Transformer model.
"""
from typing import Optional, Tuple, TYPE_CHECKING

import torch
from src.model.inference.sampling import LogitSampler

if TYPE_CHECKING:
    from src.model.model import Transformer
    from src.model.structures import SamplingConfig
    from src.model.layers.attention.kv_cache import KVCache


class SpeculativeEngine:
    """
    Handles the generation and validation of speculative chunks using a draft model.
    """

    def __init__(self, sampler: LogitSampler = None):
        self.sampler = sampler or LogitSampler()

    def generate_chunk(
        self,
        draft_model: "Transformer",
        tokens: torch.Tensor,
        speculative_steps: int,
        sampling_config: "SamplingConfig",
        draft_cache: Optional["KVCache"] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a speculative chunk of tokens using a draft model.
        """
        draft_tokens = tokens
        with torch.no_grad():
            # Initial forward pass if cache is empty or needs sync
            if draft_cache and draft_cache.current_pos > 0:
                # Synchronization logic as in previous turn
                draft_cache.rollback(1)
                draft_outputs = draft_model(tokens[:, -1:], kv_cache=draft_cache)
            else:
                # Standard prompt processing
                draft_outputs = draft_model(tokens, kv_cache=draft_cache)

            generated_chunks = []
            draft_logits = draft_outputs.logits
            for i in range(speculative_steps):
                next_token = self.sampler.sample(
                    draft_logits[:, -1, :],
                    sampling_config.temperature,
                    sampling_config.dynamic_top_k or sampling_config.top_k,
                    sampling_config.top_p,
                    sampling_config.min_p,
                    sampling_config.logit_soft_cap,
                )
                generated_chunks.append(next_token)

                if i < speculative_steps - 1:
                    draft_outputs = draft_model(next_token, kv_cache=draft_cache)
                    draft_logits = draft_outputs.logits

            draft_tokens = torch.cat([tokens] + generated_chunks, dim=1)

        return draft_tokens[:, tokens.size(1) :], draft_tokens

    def validate_chunk(
        self,
        true_logits: torch.Tensor,
        speculative_chunk: torch.Tensor,
        sampling_config: "SamplingConfig",
    ) -> torch.Tensor:
        """
        Validates a speculative chunk against logits from the main model.
        Returns the accepted prefix of the chunk.
        """
        if speculative_chunk.size(0) != 1:
            raise NotImplementedError("Speculative decoding only supports batch size 1.")

        # Batch sampling for verification
        verification_tokens = self.sampler.sample(
            true_logits.view(-1, true_logits.size(-1)),
            sampling_config.temperature,
            sampling_config.dynamic_top_k or sampling_config.top_k,
            sampling_config.top_p,
            sampling_config.min_p,
            sampling_config.logit_soft_cap,
        ).view(speculative_chunk.shape)

        mismatches = (speculative_chunk != verification_tokens).long()

        if not mismatches.any():
            return speculative_chunk

        first_mismatch_idx = torch.argmax(mismatches, dim=1)[0]

        if first_mismatch_idx == 0 and mismatches[0, 0] == 1:
            return verification_tokens[:, :1]

        accepted_prefix = speculative_chunk[:, :first_mismatch_idx]
        corrected_token = verification_tokens[:, first_mismatch_idx : first_mismatch_idx + 1]

        return torch.cat([accepted_prefix, corrected_token], dim=1)
