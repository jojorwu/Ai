"""
Implements the main text generation pipeline for the Transformer model.
"""
import logging
from typing import Generator, Tuple

import torch
from src.model.sampling import LogitSampler
from src.model.speculative import SpeculativeEngine


class TextGenerator:
    """
    Orchestrates the generation process, including KV caching and speculative decoding.
    """

    def __init__(
        self,
        model,
        sampler: LogitSampler = None,
        speculative_engine: SpeculativeEngine = None,
    ):
        self.model = model
        self.sampler = sampler or LogitSampler()
        self.speculative_engine = (
            speculative_engine or SpeculativeEngine(sampler=self.sampler)
        )

    def generate(
        self, inputs
    ) -> Generator[Tuple[torch.Tensor, float, torch.Tensor | None], None, None]:
        """
        Generates a sequence of tokens.
        """
        self.model.eval()
        draft_model = self.model.draft_model or self.model
        tokens = inputs.start_tokens.to(self.model.device)
        total_generated = 0

        # Initialize KV caches
        main_cache, draft_cache = self._prepare_caches(inputs, draft_model, tokens)

        # Initial forward pass to populate caches if needed
        last_logit, current_ltm_memory = self._initial_sync(inputs, main_cache, tokens)

        while total_generated < inputs.max_new_tokens:
            # 1. Generate speculative chunk
            speculative_chunk, _ = self.speculative_engine.generate_chunk(
                draft_model,
                tokens,
                inputs.speculative_config.speculative_steps,
                inputs.sampling_config,
                draft_cache=draft_cache,
            )
            spec_len = speculative_chunk.size(1)

            # 2. Run main model on the speculative chunk
            with torch.enable_grad():
                true_logits, value, _, current_ltm_memory = self.model(
                    speculative_chunk,
                    ltm_override=inputs.ltm_override,
                    ltm_memory=current_ltm_memory,
                    kv_cache=main_cache,
                )

            # 3. Surprise calculation (remains a model-specific detail for now)
            # We assume the model has a way to calculate surprise from its heads.
            surprise = self.model.calculate_surprise(value, inputs.ltm_override)

            # 4. Validate chunk
            all_validation_logits = torch.cat([last_logit, true_logits[:, :-1, :]], dim=1)
            accepted_chunk = self.speculative_engine.validate_chunk(
                all_validation_logits,
                speculative_chunk,
                inputs.sampling_config,
            )
            accepted_len = accepted_chunk.size(1)

            # 5. Handle acceptance and synchronization
            is_all_accepted = (accepted_len == spec_len) and torch.equal(
                accepted_chunk, speculative_chunk
            )

            if not is_all_accepted:
                # Rollback and re-process
                rollback_len = spec_len - (accepted_len - 1)
                main_cache.rollback(rollback_len)
                if draft_cache is not main_cache:
                    draft_cache.rollback(rollback_len)

                tokens = torch.cat((tokens, accepted_chunk), dim=1)
                with torch.enable_grad():
                    corrected_logits, _, _, current_ltm_memory = self.model(
                        tokens[:, -1:],
                        ltm_override=inputs.ltm_override,
                        ltm_memory=current_ltm_memory,
                        kv_cache=main_cache,
                    )
                    if draft_cache is not main_cache:
                        with torch.no_grad():
                            draft_model(tokens[:, -1:], kv_cache=draft_cache)
                    last_logit = corrected_logits[:, -1:, :]
            else:
                # Everything accepted
                tokens = torch.cat((tokens, accepted_chunk), dim=1)
                last_logit = true_logits[:, -1:, :]

            main_cache.detach()
            if draft_cache is not main_cache:
                draft_cache.detach()

            yield accepted_chunk, surprise, current_ltm_memory
            total_generated += accepted_len

    def _prepare_caches(self, inputs, draft_model, tokens):
        """Prepares or initializes KV caches for main and draft models."""
        from src.model.layers.kv_cache import KVCache, KVCacheConfig

        if inputs.kv_cache is not None:
            main_cache = inputs.kv_cache
        else:
            d_k = self.model.config.model.d_model // self.model.config.model.num_heads
            main_cache = KVCache(
                KVCacheConfig(
                    num_layers=self.model.config.model.num_layers,
                    batch_size=tokens.shape[0],
                    num_kv_heads=self.model.config.model.num_kv_heads,
                    d_k=d_k,
                    max_seq_len=self.model.config.model.max_seq_len,
                ),
                device=self.model.device,
                dtype=self.model.layers.embedding.weight.dtype,
            )

        if inputs.draft_cache is not None:
            draft_cache = inputs.draft_cache
        elif draft_model is not self.model:
            draft_cache = KVCache(
                KVCacheConfig(
                    num_layers=draft_model.config.model.num_layers,
                    batch_size=tokens.shape[0],
                    num_kv_heads=draft_model.config.model.num_kv_heads,
                    d_k=draft_model.config.model.d_model // draft_model.config.model.num_heads,
                    max_seq_len=draft_model.config.model.max_seq_len,
                ),
                device=self.model.device,
                dtype=self.model.layers.embedding.weight.dtype,
            )
        else:
            draft_cache = main_cache

        return main_cache, draft_cache

    def _initial_sync(self, inputs, main_cache, tokens):
        """Performs initial forward pass to synchronize the main cache."""
        current_ltm_memory = inputs.ltm_memory
        if main_cache.current_pos == 0:
            with torch.no_grad():
                initial_logits, _, _, current_ltm_memory = self.model(
                    tokens[:, -self.model.config.model.max_seq_len :],
                    ltm_override=inputs.ltm_override,
                    ltm_memory=current_ltm_memory,
                    kv_cache=main_cache,
                )
                last_logit = initial_logits[:, -1:, :]
        else:
            # Assume cache is already in sync with prompt
            with torch.no_grad():
                # We need the last logit from the current cache state
                # The easiest way is to run a small forward pass on the last token.
                # But wait, if the cache is already at current_pos, we need to rollback by 1
                # to get the logits for that last token without duplicating it.
                main_cache.rollback(1)
                initial_logits, _, _, current_ltm_memory = self.model(
                    tokens[:, -1:],
                    ltm_override=inputs.ltm_override,
                    ltm_memory=current_ltm_memory,
                    kv_cache=main_cache,
                )
                last_logit = initial_logits[:, -1:, :]
        return last_logit, current_ltm_memory
