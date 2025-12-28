"""
PyTorch implementation of the main Transformer model.
"""
import copy
import math
from dataclasses import dataclass
from typing import Generator, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from bitsandbytes.nn import Linear4bit

from config import DecoderBlockConfig, TransformerConfig
from nn_components.decoder_block import DecoderBlock, ForwardPassInput
from nn_components.embedding import Embedding
from nn_components.linear import Linear
from nn_components.long_term_memory import LongTermMemory
from nn_components.rms_norm import RMSNorm
from nn_components.rotary_embedding import precompute_rope_embeddings
from nn_components.value_head import ValueHead


@dataclass
class GenerateInput:
    """Dataclass for storing inputs to the generate method."""
    start_tokens: torch.Tensor
    max_new_tokens: int
    temperature: float = 1.0
    top_k: int = 0
    speculative_steps: int = 5
    value_threshold: float = -1.0
    max_retries: int = 3
    dynamic_top_k: int = None

class Transformer(nn.Module):
    """
    Full GPT-style (decoder-only) Transformer model, migrated to PyTorch.
    """
    _no_split_modules = ["DecoderBlock"]

    def __init__(self, config: TransformerConfig, load_in_4bit: bool = False):
        super().__init__()
        self.config = config
        self.embedding = Embedding(config.vocab_size, config.model.d_model)
        self.long_term_memory = self._init_ltm(config)
        self.rope_cos, self.rope_sin = self._init_rope_embeddings(config)
        self.decoder_blocks, self.final_norm, self.value_head = self._init_decoder(
            config, load_in_4bit
        )
        self.embedding.weights = self.embedding.embedding.weight

    def _init_ltm(self, config: TransformerConfig):
        if config.model.ltm_d_hidden and config.model.ltm_num_layers:
            return LongTermMemory(
                d_model=config.model.d_model,
                d_hidden=config.model.ltm_d_hidden,
                num_layers=config.model.ltm_num_layers
            )
        return None

    def _init_rope_embeddings(self, config: TransformerConfig):
        d_k = config.model.d_model // config.model.num_heads
        rope_cos, rope_sin = precompute_rope_embeddings(
            d_k, config.model.max_seq_len
        )
        self.register_buffer("rope_cos_buf", rope_cos)
        self.register_buffer("rope_sin_buf", rope_sin)
        return self.rope_cos_buf, self.rope_sin_buf

    def _init_decoder(self, config: TransformerConfig, load_in_4bit: bool):
        block_config = self._create_block_config(load_in_4bit)
        decoder_blocks = nn.ModuleList(
            [DecoderBlock(block_config) for _ in range(config.model.num_layers)]
        )
        final_norm = RMSNorm(config.model.d_model)
        linear_class = Linear4bit if load_in_4bit else Linear
        value_head = ValueHead(config.model.d_model, linear_class=linear_class)
        return decoder_blocks, final_norm, value_head

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
            load_in_4bit=load_in_4bit)

    def forward(
        self, x: torch.Tensor, ltm_state: torch.Tensor = None, dynamic_top_k: int = None
    ):
        """Forward pass of the model."""
        h = self.embedding(x) * math.sqrt(self.config.model.d_model)
        if ltm_state is None:
            if self.long_term_memory:
                ltm_state = self.long_term_memory(h.mean(dim=1, keepdim=True))
            else:
                ltm_state = torch.zeros(
                    (h.size(0), 1, h.size(2)), device=h.device, dtype=h.dtype
                )

        total_aux_loss = torch.tensor(0.0, device=x.device)
        for i, block in enumerate(self.decoder_blocks):
            inputs = ForwardPassInput(
                x=h, ltm_state=ltm_state, layer_idx=i, dynamic_top_k=dynamic_top_k
            )
            h, aux_loss = block(inputs)
            if aux_loss is not None:
                total_aux_loss += aux_loss

        h = self.final_norm(h)
        logits = F.linear(h, self.embedding.weights)
        value = self.value_head(h[:, -1, :])
        return logits, value, total_aux_loss

    def _sample_from_logits(self, logits, temperature, top_k):
        """Samples a token from logits."""
        if temperature == 0.0:
            _, next_token = torch.topk(logits, k=1, dim=-1)
        else:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1, None]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        return next_token

    def _calculate_surprise(self, value: torch.Tensor) -> float:
        """Calculates 'surprise' by backpropagating value and getting LTM grad norm."""
        if not self.long_term_memory or not value.requires_grad:
            return 0.0

        self.long_term_memory.zero_grad()
        value.backward(retain_graph=True)
        grad_tensors = [
            p.grad.detach() for p in self.long_term_memory.parameters() if p.grad is not None
        ]
        if not grad_tensors:
            return 0.0

        surprise = torch.norm(torch.cat([t.flatten() for t in grad_tensors])).item()
        self.long_term_memory.zero_grad()
        return surprise

    def _generate_speculative_chunk(self, draft_model, tokens, inputs):
        """Generates a speculative chunk of tokens."""
        draft_tokens = tokens
        with torch.no_grad():
            for _ in range(inputs.speculative_steps):
                draft_logits, _, _ = draft_model(
                    draft_tokens[:, -self.config.model.max_seq_len:]
                )
                next_token = self._sample_from_logits(
                    draft_logits[:, -1, :],
                    inputs.temperature,
                    inputs.dynamic_top_k or inputs.top_k
                )
                draft_tokens = torch.cat((draft_tokens, next_token), dim=1)
        return draft_tokens[:, tokens.size(1):], draft_tokens

    def _validate_and_accept_chunk(self, true_logits, speculative_chunk, inputs):
        """Validates the speculative chunk and returns the accepted tokens."""
        accepted_tokens = []
        for i in range(speculative_chunk.size(1)):
            true_next_token_logits = true_logits[:, i, :]
            draft_token = speculative_chunk[:, i].unsqueeze(-1)
            resampled_token = self._sample_from_logits(
                true_next_token_logits,
                inputs.temperature,
                inputs.dynamic_top_k or inputs.top_k
            )
            if resampled_token.item() == draft_token.item():
                accepted_tokens.append(draft_token)
            else:
                accepted_tokens.append(resampled_token)
                break
        return torch.cat(accepted_tokens, dim=1) if accepted_tokens else None

    def generate(self, inputs: GenerateInput) -> Generator[Tuple[torch.Tensor, float], None, None]:
        """
        Generates a sequence of tokens using speculative decoding.
        Yields chunks of accepted tokens and the surprise value.
        """
        self.eval()
        tokens = inputs.start_tokens
        total_generated = 0
        draft_model = copy.deepcopy(self)

        while total_generated < inputs.max_new_tokens:
            speculative_chunk, draft_tokens = self._generate_speculative_chunk(
                draft_model, tokens, inputs
            )
            with torch.enable_grad():
                true_logits, value, _ = self(draft_tokens[:, -self.config.model.max_seq_len:])
            surprise = self._calculate_surprise(value)

            accepted_chunk = self._validate_and_accept_chunk(
                true_logits, speculative_chunk, inputs
            )

            if accepted_chunk is not None:
                yield accepted_chunk, surprise
                tokens = torch.cat((tokens, accepted_chunk), dim=1)
                total_generated += accepted_chunk.size(1)
            else:
                logits, _, _ = self(tokens[:, -self.config.model.max_seq_len:])
                next_token = self._sample_from_logits(
                    logits[:, -1, :],
                    inputs.temperature,
                    inputs.dynamic_top_k or inputs.top_k
                )
                yield next_token, surprise
                tokens = torch.cat((tokens, next_token), dim=1)
                total_generated += 1

    @property
    def device(self):
        """Returns the device of the model's embedding layer."""
        return self.embedding.embedding.weight.device

    def count_parameters(self):
        """Counts the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
