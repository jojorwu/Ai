"""
Implements the main text generation pipeline for the Transformer model.
"""
from __future__ import annotations
import logging
from typing import Generator, Tuple, TYPE_CHECKING

import torch

from src.model.inference.sampling import LogitSampler
from src.model.inference.speculative import SpeculativeEngine
from src.model.structures import GenerationResult
from src.utils.exceptions import GenerationError

if TYPE_CHECKING:
    from src.model.model import Transformer
    from src.model.structures import GenerateInput
    from src.model.layers.attention.kv_cache import KVCache, KVCacheConfig


class TextGenerator:
    """
    Orchestrates the generation process.

    This includes handling KV caching, speculative decoding, and surprise
    calculation to manage the generation lifecycle.
    """

    def __init__(
        self,
        model: Transformer,
        sampler: LogitSampler | None = None,
        speculative_engine: SpeculativeEngine | None = None,
    ) -> None:
        """
        Initializes the TextGenerator.

        Args:
            model: The Transformer model to use for generation.
            sampler: Optional LogitSampler for token selection.
            speculative_engine: Optional SpeculativeEngine for acceleration.
        """
        self.model = model
        self.sampler = sampler or LogitSampler()
        self.speculative_engine = (
            speculative_engine or SpeculativeEngine(sampler=self.sampler)
        )

    def generate(
        self, inputs: GenerateInput
    ) -> Generator[GenerationResult, None, None]:
        """
        Generates a sequence of tokens.

        Args:
            inputs: Configuration and initial state for generation.

        Yields:
            GenerationResult objects containing tokens, surprise, and LTM memory.
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
            if main_cache.current_pos >= main_cache.config.max_seq_len:
                raise GenerationError(
                    f"Maximum sequence length ({main_cache.config.max_seq_len}) reached."
                )

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
                outputs = self.model(
                    speculative_chunk,
                    ltm_override=inputs.ltm_override,
                    ltm_memory=current_ltm_memory,
                    kv_cache=main_cache,
                )
                current_ltm_memory = outputs.ltm_memory

            # 3. Surprise calculation (remains a model-specific detail for now)
            surprise = self.model.calculate_surprise(outputs.value, inputs.ltm_override)

            # 4. Validate chunk
            all_validation_logits = torch.cat(
                [last_logit, outputs.logits[:, :-1, :]], dim=1
            )
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
                    corrected_outputs = self.model(
                        tokens[:, -1:],
                        ltm_override=inputs.ltm_override,
                        ltm_memory=current_ltm_memory,
                        kv_cache=main_cache,
                    )
                    current_ltm_memory = corrected_outputs.ltm_memory
                    if draft_cache is not main_cache:
                        with torch.no_grad():
                            draft_model(tokens[:, -1:], kv_cache=draft_cache)
                    last_logit = corrected_outputs.logits[:, -1:, :]
            else:
                # Everything accepted
                tokens = torch.cat((tokens, accepted_chunk), dim=1)
                last_logit = outputs.logits[:, -1:, :]

            main_cache.detach()
            if draft_cache is not main_cache:
                draft_cache.detach()

            yield GenerationResult(
                tokens=accepted_chunk,
                surprise=surprise,
                ltm_memory=current_ltm_memory
            )
            total_generated += accepted_len

            # Check for stop tokens
            if inputs.stop_tokens:
                stop_found = False
                for token_id in inputs.stop_tokens:
                    if (accepted_chunk == token_id).any():
                        stop_found = True
                        break
                if stop_found:
                    logging.info("Stop token encountered. Terminating generation.")
                    break

    def _prepare_caches(
        self, inputs: GenerateInput, draft_model: Transformer, tokens: torch.Tensor
    ) -> Tuple[KVCache, KVCache]:
        """
        Prepares or initializes KV caches for main and draft models.

        Args:
            inputs: Generation inputs containing optional existing caches.
            draft_model: The draft model to prepare a cache for.
            tokens: Initial tokens to determine batch size.

        Returns:
            A tuple of (main_cache, draft_cache).
        """
        from src.model.layers.attention.kv_cache import KVCache, KVCacheConfig

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

    def _initial_sync(
        self, inputs: GenerateInput, main_cache: KVCache, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """
        Performs initial forward pass to synchronize the main cache with the prompt.

        Args:
            inputs: Generation inputs.
            main_cache: The main KV cache to synchronize.
            tokens: Initial prompt tokens.

        Returns:
            A tuple containing the last logit and current LTM memory.
        """
        current_ltm_memory = inputs.ltm_memory
        if main_cache.current_pos == 0:
            with torch.no_grad():
                # Process the entire prompt to ensure LTM and KV cache are fully synchronized.
                outputs = self.model(
                    tokens,
                    ltm_override=inputs.ltm_override,
                    ltm_memory=current_ltm_memory,
                    kv_cache=main_cache,
                    images=inputs.images,
                )
                current_ltm_memory = outputs.ltm_memory
                last_logit = outputs.logits[:, -1:, :]
        else:
            # Assume cache is already in sync with prompt
            with torch.no_grad():
                # We need the last logit from the current cache state
                main_cache.rollback(1)
                outputs = self.model(
                    tokens[:, -1:],
                    ltm_override=inputs.ltm_override,
                    ltm_memory=current_ltm_memory,
                    kv_cache=main_cache,
                )
                current_ltm_memory = outputs.ltm_memory
                last_logit = outputs.logits[:, -1:, :]
        return last_logit, current_ltm_memory
