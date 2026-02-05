"""
PyTorch implementation of the main Transformer model.
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Generator, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.model.layers.attention.kv_cache import KVCache
from torch import nn
from torch.nn import functional as F

from src.config.core import TransformerConfig
from src.utils.exceptions import MultimodalError
from src.model.inference.generation import GenerationMixin
from src.model.inference.generator import TextGenerator
from src.model.initializer import ModelInitializer
from src.model.titans.titans_engine import TitansForwardEngine
from src.model.structures import (
    ForwardOutput,
    GenerateInput,
    GenerationResult,
    ModelLayers,
    RopeEmbeddings,
    SamplingConfig,
    SpeculativeConfig,
)
from src.model.layers.blocks.decoder_block import DecoderBlock
from src.model.layers.core.embedding import Embedding
from src.model.layers.titans.gating import GatingNetwork
from src.model.layers.titans.long_term_memory import LongTermMemory
from src.model.layers.core.rms_norm import RMSNorm
from src.model.layers.heads.value_head import ValueHead


class Transformer(nn.Module, GenerationMixin):
    """
    A decoder-only Transformer model with a dual-head architecture.

    It supports dynamic layer skipping and Mixture of Experts (MoE) allocation
    based on a complexity score from its Long-Term Memory (LTM). The model
    is designed for agent-based learning and supports speculative decoding
    to accelerate generation.
    """

    _no_split_modules = ["DecoderBlock"]

    def __init__(self, config: TransformerConfig, load_in_4bit: bool = False) -> None:
        """
        Initializes the Transformer model.

        Args:
            config: Configuration object for the Transformer.
            load_in_4bit: Whether to load the model in 4-bit precision.
        """
        super().__init__()
        self.config = config
        self.load_in_4bit = load_in_4bit

        self.initializer = ModelInitializer(self)
        self.rope_embeddings = self.initializer.init_rope_embeddings()
        self.layers = self.initializer.init_layers()
        self._draft_model: Transformer | None = None

        # Weight tying: share weights between embedding and policy head
        self.layers.embedding.weight = self.layers.embedding.embedding.weight

        # Core engines
        self.generator = TextGenerator(self)
        self.titans_engine = TitansForwardEngine(self)

    def count_parameters(self) -> int:
        """
        Counts the number of trainable parameters in the model.

        Returns:
            The total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,
        ltm_state: torch.Tensor | None = None,
        ltm_memory: torch.Tensor | None = None,
        dynamic_top_k: int | None = None,
        ltm_override: nn.Module | None = None,
        kv_cache: KVCache | None = None,
        images: torch.Tensor | None = None,
    ) -> ForwardOutput:
        """
        Performs the forward pass of the Transformer model.

        Args:
            x: Input token IDs of shape [batch, seq_len].
            ltm_state: Optional pre-calculated Long-Term Memory state.
            ltm_memory: Optional persistent LTM associative matrix.
            dynamic_top_k: Optional override for MoE top_k.
            ltm_override: Optional LTM module to use instead of the model's own.
            kv_cache: Optional persistent Key-Value cache for inference.
            images: Optional input images of shape [batch, channels, height, width].

        Returns:
            A ForwardOutput dataclass containing logits, value, and memory states.
        """
        # 1. Embed input sequence
        h = self.layers.embedding(x) * math.sqrt(self.config.model.d_model)

        # Apply stable embedding normalization
        h = self.layers.post_embedding_norm(h)

        # 2. Integrate image embeddings if provided
        if images is not None:
            if self.layers.vision_encoder is None:
                raise MultimodalError(
                    "Images provided but VisionEncoder is not initialized."
                )
            # ONLY prepend image embeddings if they are NOT already in the cache.
            # This prevents redundant vision processing during auto-regressive generation.
            if kv_cache is None or kv_cache.current_pos == 0:
                image_embeds = self.layers.vision_encoder(images)
                # Apply stable embedding normalization to vision tokens as well
                image_embeds = self.layers.post_embedding_norm(image_embeds)
                # Prepend image embeddings to text embeddings
                h = torch.cat([image_embeds, h], dim=1)

        # 3. Delegate sequence processing to the TitansForwardEngine
        h, total_aux_loss, new_ltm_memory = self.titans_engine.process_sequence(
            h=h,
            ltm_state=ltm_state,
            ltm_memory=ltm_memory,
            dynamic_top_k=dynamic_top_k,
            ltm_override=ltm_override,
            kv_cache=kv_cache,
        )

        # 4. Final normalization and head projections.
        h = self.layers.final_norm(h)
        # Policy head: projects the final hidden states to the vocabulary size.
        # We only take the logits for the text tokens if images were prepended?
        # Actually, standard multimodal LLMs provide logits for all positions.
        logits = F.linear(  # pylint: disable=not-callable
            h, self.layers.embedding.weight
        )

        # Apply logit soft-clamping if configured
        if self.config.model.logit_soft_cap is not None:
            logits = self.config.model.logit_soft_cap * torch.tanh(
                logits / self.config.model.logit_soft_cap
            )
        # Value head: projects the final hidden state of the last token to a
        # single scalar value, predicting the "usefulness" of the sequence.
        value = self.layers.value_head(h[:, -1, :])

        # Increment the KV cache position if it's being used.
        if kv_cache is not None:
            kv_cache.increment_pos(h.shape[1])

        return ForwardOutput(
            logits=logits,
            value=value,
            aux_loss=total_aux_loss,
            ltm_memory=new_ltm_memory,
        )

    def generate(
        self, inputs: GenerateInput
    ) -> Generator[GenerationResult, None, None]:
        """
        Generates a sequence of tokens.

        Args:
            inputs: Input parameters for the generation process.

        Yields:
            GenerationResult objects containing tokens and metadata.
        """
        return self.generator.generate(inputs)

    def train(self, mode: bool = True) -> Transformer:
        """
        Sets the model and its draft model to training mode.

        Args:
            mode: Whether to set training mode (True) or evaluation mode (False).

        Returns:
            The model instance.
        """
        super().train(mode)
        if self._draft_model:
            self._draft_model.train(mode)
        return self

    def eval(self) -> Transformer:
        """
        Sets the model and its draft model to evaluation mode.

        Returns:
            The model instance.
        """
        super().eval()
        if self._draft_model:
            self._draft_model.eval()
        return self

    @property
    def draft_model(self) -> Transformer | None:
        """
        Lazy-initializes the draft model for speculative decoding.

        Returns:
            The draft model instance or None if not applicable.
        """
        if self._draft_model is None:
            self._draft_model = self.initializer.init_draft_model()
        return self._draft_model

    @property
    def device(self) -> torch.device:
        """
        Returns the device where the model's parameters are located.

        Returns:
            The torch.device of the embedding layer.
        """
        return self.layers.embedding.embedding.weight.device
