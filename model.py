"""
PyTorch implementation of the main Transformer model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Generator, Tuple

import bitsandbytes as bnb
from accelerate import init_empty_weights
from bitsandbytes.nn import Linear4bit

from config import TransformerConfig, DecoderBlockConfig
from nn_components.decoder_block import DecoderBlock, ForwardPassInput
from nn_components.embedding import Embedding
from nn_components.linear import Linear
from nn_components.rms_norm import RMSNorm
from nn_components.rotary_embedding import precompute_rope_embeddings
from nn_components.activations import Tanh
from nn_components.long_term_memory import LongTermMemory

@dataclass
class GenerateInput:
    """Dataclass for storing inputs to the generate method."""
    start_tokens: torch.Tensor
    max_new_tokens: int
    temperature: float = 1.0
    top_k: int = 0

class Transformer(nn.Module):
    """
    Full GPT-style (decoder-only) Transformer model, migrated to PyTorch.
    """
    _no_split_modules = ["DecoderBlock"]

    def __init__(self, config: TransformerConfig, load_in_4bit: bool = False):
        super().__init__()
        self.config = config
        self._initialize_layers(config, load_in_4bit)

    def _initialize_layers(self, config: TransformerConfig, load_in_4bit: bool):
        """Initializes the layers of the model."""
        d_k = config.model.d_model // config.model.num_heads
        rope_cos, rope_sin = precompute_rope_embeddings(d_k, config.model.max_seq_len)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        self.embedding = Embedding(config.vocab_size, config.model.d_model)

        if config.model.ltm_d_hidden and config.model.ltm_num_layers:
            self.long_term_memory = LongTermMemory(
                d_model=config.model.d_model,
                d_hidden=config.model.ltm_d_hidden,
                num_layers=config.model.ltm_num_layers
            )
        else:
            self.long_term_memory = None

        # self.vision_encoder = VisionEncoder(...)

        block_config = self._create_block_config(load_in_4bit)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(block_config) for _ in range(config.model.num_layers)]
        )
        self.final_norm = RMSNorm(config.model.d_model)

        # Dual-head architecture
        linear_class = Linear4bit if load_in_4bit else Linear
        self.value_head_linear = linear_class(config.model.d_model, 1, bias=False)
        self.value_head_activation = Tanh()

        # Weight tying for the policy head
        self.embedding.weights.data = self.embedding.embedding.weight

    def _create_block_config(self, load_in_4bit: bool) -> DecoderBlockConfig:
        """Helper method to create the DecoderBlockConfig."""
        return DecoderBlockConfig(
            d_model=self.config.model.d_model,
            num_heads=self.config.model.num_heads,
            d_ff=self.config.model.d_ff,
            dropout_rate=self.config.model.dropout_rate,
            num_kv_heads=self.config.model.num_kv_heads,
            rotary_emb=(self.rope_cos, self.rope_sin),
            num_layers=self.config.model.num_layers,
            long_term_memory=self.long_term_memory,
            num_experts=self.config.model.num_experts,
            top_k_experts=self.config.model.top_k_experts,
            load_in_4bit=load_in_4bit
        )

    def forward(self, x: torch.Tensor, ltm_state: torch.Tensor = None, dynamic_top_k: int = None):
        """Performs the forward pass of the model."""
        h = self.embedding(x) * math.sqrt(self.config.model.d_model)

        if self.long_term_memory:
            # LTM processes the mean of embeddings to generate context
            ltm_input = h.mean(dim=1, keepdim=True)
            memory_context = self.long_term_memory(ltm_input)
            h = torch.cat([memory_context.expand(-1, h.size(1), -1), h], dim=1)

        if ltm_state is None and self.long_term_memory:
             ltm_state = self.long_term_memory(h.mean(dim=1, keepdim=True))
        elif ltm_state is None:
            ltm_state = torch.zeros_like(h[:, :1, :])


        total_aux_loss = torch.tensor(0.0, device=x.device)
        for i, block in enumerate(self.decoder_blocks):
            inputs = ForwardPassInput(
                x=h,
                ltm_state=ltm_state,
                layer_idx=i,
                dynamic_top_k=dynamic_top_k
            )
            h, aux_loss = block(inputs)
            total_aux_loss += aux_loss

        h = self.final_norm(h)

        # Policy head (logits) using tied weights
        logits = F.linear(h, self.embedding.weights)

        # Value head
        last_token_hidden_state = h[:, -1, :]
        value_hidden = self.value_head_linear(last_token_hidden_state)
        value = self.value_head_activation(value_hidden)

        return logits, value, total_aux_loss

    @torch.no_grad()
    def generate(self, inputs: GenerateInput) -> torch.Tensor:
        """
        Generates a sequence of tokens. Simplified for migration.
        This is a basic auto-regressive generation loop.
        """
        self.eval()
        tokens = inputs.start_tokens

        for _ in range(inputs.max_new_tokens):
            # Crop context if it exceeds max sequence length
            tokens_cond = tokens if tokens.size(1) <= self.config.model.max_seq_len else tokens[:, -self.config.model.max_seq_len:]

            logits, _, _ = self(tokens_cond)

            # Get logits for the last token
            logits = logits[:, -1, :]

            if inputs.temperature == 0.0:
                _, next_token = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / inputs.temperature
                if inputs.top_k > 0:
                    top_k = torch.topk(logits, k=inputs.top_k, dim=-1)
                    logits[logits < top_k.values[:, -1, None]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat((tokens, next_token), dim=1)

        return tokens
