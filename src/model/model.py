"""
PyTorch implementation of the main Transformer model.
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Generator, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from src.config.core import TransformerConfig
from src.model.generation import GenerationMixin
from src.model.generator import TextGenerator
from src.model.initializer import ModelInitializer
from src.model.titans_engine import TitansForwardEngine
from src.model.structures import (
    GenerateInput,
    ModelLayers,
    RopeEmbeddings,
    SamplingConfig,
    SpeculativeConfig,
)
from src.model.layers.decoder_block import DecoderBlock
from src.model.layers.embedding import Embedding
from src.model.layers.gating import GatingNetwork
from src.model.layers.long_term_memory import LongTermMemory
from src.model.layers.rms_norm import RMSNorm
from src.model.layers.value_head import ValueHead


class Transformer(nn.Module, GenerationMixin):
    """
    A decoder-only Transformer model with a dual-head architecture for policy and
    value prediction. It supports dynamic layer skipping and Mixture of Experts (MoE)
    allocation based on a complexity score from its Long-Term Memory (LTM).

    The model is designed for agent-based learning and supports speculative decoding
    to accelerate generation.
    """

    _no_split_modules = ["DecoderBlock"]

    def __init__(self, config: TransformerConfig, load_in_4bit: bool = False):
        super().__init__()
        self.config = config
        self._config = config  # Keep a copy of the full config
        self.load_in_4bit = load_in_4bit

        initializer = ModelInitializer(self)
        self.rope_embeddings = initializer.init_rope_embeddings()
        self.layers = initializer.init_layers()
        self.draft_model = initializer.init_draft_model()

        # Weight tying: share weights between embedding and policy head
        self.layers.embedding.weight = self.layers.embedding.embedding.weight

        # Core engines
        self.generator = TextGenerator(self)
        self.titans_engine = TitansForwardEngine(self)

    def count_parameters(self):
        """Counts the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,
        ltm_state: torch.Tensor = None,
        dynamic_top_k: int = None,
        ltm_override: nn.Module | None = None,
        kv_cache: "KVCache" = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass, delegating Titans logic to TitansForwardEngine.
        """
        # 1. Embed input sequence
        h = self.layers.embedding(x) * math.sqrt(self.config.model.d_model)

        # 2. Delegate sequence processing to the TitansForwardEngine
        h, total_aux_loss = self.titans_engine.process_sequence(
            h=h,
            ltm_state=ltm_state,
            dynamic_top_k=dynamic_top_k,
            ltm_override=ltm_override,
            kv_cache=kv_cache,
        )

        # 3. Final normalization and head projections.
        h = self.layers.final_norm(h)
        # Policy head: projects the final hidden states to the vocabulary size
        # to get logits. It shares weights with the embedding layer (weight tying).
        logits = F.linear(  # pylint: disable=not-callable
            h, self.layers.embedding.weight
        )
        # Value head: projects the final hidden state of the last token to a
        # single scalar value, predicting the "usefulness" of the sequence.
        value = self.layers.value_head(h[:, -1, :])

        # Increment the KV cache position if it's being used.
        if kv_cache is not None:
            kv_cache.increment_pos(x.shape[1])

        return logits, value, total_aux_loss

    def generate(
        self, inputs: GenerateInput
    ) -> Generator[Tuple[torch.Tensor, float], None, None]:
        """
        Generates a sequence of tokens, delegating to the TextGenerator pipeline.
        """
        return self.generator.generate(inputs)

    def train(self, mode: bool = True):
        """
        Overrides the default `train` method to also set the mode for the draft model.
        """
        super().train(mode)
        if self.draft_model:
            self.draft_model.train(mode)
        return self

    def eval(self):
        """
        Overrides the default `eval` method to also set the mode for the draft model.
        """
        super().eval()
        if self.draft_model:
            self.draft_model.eval()
        return self

    @property
    def device(self):
        """Returns the device of the model's embedding layer."""
        return self.layers.embedding.embedding.weight.device
