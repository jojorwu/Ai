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
from src.model.structures import (
    GenerateInput,
    ModelLayers,
    RopeEmbeddings,
    SamplingConfig,
    SpeculativeConfig,
)
from src.model.layers.decoder_block import DecoderBlock, ForwardPassInput
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

        # Generation pipeline
        self.generator = TextGenerator(self)

    def count_parameters(self):
        """Counts the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(  # pylint: disable=too-many-locals
        self,
        x: torch.Tensor,
        ltm_state: torch.Tensor = None,
        dynamic_top_k: int = None,  # This is passed in from generation config
        ltm_override: nn.Module | None = None,
        kv_cache: 'KVCache' = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the Transformer model.

        This method implements the core logic of the "Titans" architecture, where
        a GatingNetwork dynamically adjusts the model's depth and expert allocation
        based on a compressed representation of the sequence from the Long-Term Memory (LTM).
        This allows the model to allocate fewer resources for simpler tokens and
        more for complex ones, optimizing computational efficiency.

        Args:
            x: Input tensor of token IDs. Shape: (batch_size, seq_len).
            ltm_state: An optional pre-computed state from the Long-Term Memory.
                       If None, it will be computed from the input `x`. This is useful
                       for generation where the LTM state can be cached.
            dynamic_top_k: An optional integer to override the MoE k-value determined
                           by the GatingNetwork. Used during generation with the
                           ComplexityManager.
            ltm_override: An optional LTM module to use instead of the model's default.
                          This is a key mechanism for the agent-based evolutionary
                          training, allowing each agent to have its unique LTM while
                          sharing the base model's weights.

        Returns:
            A tuple containing:
            - logits: The output logits for next token prediction.
                      Shape: (batch_size, seq_len, vocab_size).
            - value: The predicted value from the value head. Shape: (batch_size, 1).
            - total_aux_loss: The auxiliary load balancing loss from the MoE layers.
        """
        # 1. Get token embeddings. The embedding output is scaled by the square
        # root of the model dimension, a standard practice in Transformers to
        # preserve variance.
        h = self.layers.embedding(x) * math.sqrt(self.config.model.d_model)

        # 2. Determine and compute the Long-Term Memory (LTM) state.
        # An agent can override the base model's LTM with its own specialized one.
        long_term_memory = ltm_override or self.layers.long_term_memory
        if ltm_state is None:
            if long_term_memory:
                # If no LTM state is provided, compute it from the mean of the
                # input sequence embeddings. This provides a compressed summary.
                ltm_state, _ = long_term_memory(
                    h.mean(dim=1, keepdim=True)  # Use mean of sequence as summary
                )
            else:
                # If there's no LTM, use a zero tensor as a placeholder to ensure
                # the architecture remains consistent.
                ltm_state = torch.zeros(
                    (h.size(0), 1, h.size(2)), device=h.device, dtype=h.dtype
                )

        # 3. Get dynamic parameters from the GatingNetwork. This is the core of
        # the dynamic architecture. The GatingNetwork, a small neural network,
        # takes the LTM state and decides how many decoder layers and MoE experts
        # to use for this specific forward pass.
        active_layers_tensor, moe_top_k = self.layers.gating_network(ltm_state)
        # For the decoder loop, we need a single integer. Since generation has a
        # batch size of 1, taking the max works for both training and inference.
        active_layers = int(torch.max(active_layers_tensor).item())

        # The generation configuration (via ComplexityManager) can override the
        # dynamically selected top-k value for MoE.
        if dynamic_top_k is not None:
            final_dynamic_top_k = dynamic_top_k
        elif isinstance(moe_top_k, torch.Tensor):
            final_dynamic_top_k = int(moe_top_k.max().item())
        else:
            final_dynamic_top_k = moe_top_k

        # 4. Pass through the dynamically selected number of decoder blocks.
        # The model only computes the number of layers determined by the GatingNetwork,
        # saving significant computation on simpler tokens.
        total_aux_loss = torch.tensor(0.0, device=x.device)
        for i in range(active_layers):
            block = self.layers.decoder[i]
            block_input = ForwardPassInput(
                x=h,
                ltm_state=ltm_state,
                kv_cache=kv_cache,
                layer_idx=i,
                dynamic_top_k=final_dynamic_top_k,
            )
            # Gradient checkpointing is used to save memory during training by
            # recomputing activations in the backward pass instead of storing them.
            if self.config.model.gradient_checkpointing and self.training:
                h, aux_loss = torch.utils.checkpoint.checkpoint(
                    block, block_input, use_reentrant=False
                )
            else:
                h, aux_loss = block(block_input)

            if aux_loss is not None:
                total_aux_loss += aux_loss

        # 5. Final normalization and head projections.
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
