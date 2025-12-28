"""
PyTorch implementation of the main Transformer model.
"""
import copy
import math
from dataclasses import dataclass, field
from typing import Generator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DecoderBlockConfig, TransformerConfig
from nn_components.activations import Tanh
from nn_components.decoder_block import DecoderBlock, ForwardPassInput
from nn_components.embedding import Embedding
from nn_components.linear import Linear
from nn_components.long_term_memory import LongTermMemory
from nn_components.rms_norm import RMSNorm
from nn_components.rotary_embedding import precompute_rope_embeddings


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
                d_model=config.model.d_model, d_hidden=config.model.ltm_d_hidden,
                num_layers=config.model.ltm_num_layers
            )
        else:
            self.long_term_memory = None
        block_config = self._create_block_config(load_in_4bit)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(block_config) for _ in range(config.model.num_layers)])
        self.final_norm = RMSNorm(config.model.d_model)
        linear_class = Linear4bit if load_in_4bit else Linear
        self.value_head_linear = linear_class(config.model.d_model, 1, bias=False)
        self.value_head_activation = Tanh()
        self.embedding.weights = self.embedding.embedding.weight

    def _create_block_config(self, load_in_4bit: bool) -> DecoderBlockConfig:
        """Helper method to create the DecoderBlockConfig."""
        # ... (implementation is the same as before)
        return DecoderBlockConfig(
            d_model=self.config.model.d_model, num_heads=self.config.model.num_heads,
            d_ff=self.config.model.d_ff, dropout_rate=self.config.model.dropout_rate,
            num_kv_heads=self.config.model.num_kv_heads, rotary_emb=(self.rope_cos, self.rope_sin),
            num_layers=self.config.model.num_layers, long_term_memory=self.long_term_memory,
            num_experts=self.config.model.num_experts, top_k_experts=self.config.model.top_k_experts,
            load_in_4bit=load_in_4bit
        )

    def forward(self, x: torch.Tensor, ltm_state: torch.Tensor = None, dynamic_top_k: int = None):
        h = self.embedding(x) * math.sqrt(self.config.model.d_model)
        if ltm_state is None:
            ltm_state = self.long_term_memory(h.mean(dim=1, keepdim=True)) if self.long_term_memory else \
                torch.zeros((h.size(0), 1, h.size(2)), device=h.device, dtype=h.dtype)

        total_aux_loss = torch.tensor(0.0, device=x.device)
        for i, block in enumerate(self.decoder_blocks):
            inputs = ForwardPassInput(x=h, ltm_state=ltm_state, layer_idx=i, dynamic_top_k=dynamic_top_k)
            h, aux_loss = block(inputs)
            if aux_loss is not None: total_aux_loss += aux_loss

        h = self.final_norm(h)
        logits = F.linear(h, self.embedding.weights)
        value_hidden = self.value_head_linear(h[:, -1, :])
        value = self.value_head_activation(value_hidden)
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

        # Ensure LTM grads are zero before backward pass
        self.long_term_memory.zero_grad()

        # Backward pass from the value head to calculate gradients for LTM
        value.backward(retain_graph=True)

        grad_tensors = [p.grad.detach() for p in self.long_term_memory.parameters() if p.grad is not None]
        if not grad_tensors:
            return 0.0

        surprise = torch.linalg.norm(torch.cat([t.flatten() for t in grad_tensors])).item()

        # Clean up gradients
        self.long_term_memory.zero_grad()
        return surprise

    def generate(self, inputs: GenerateInput) -> Generator[Tuple[torch.Tensor, float], None, None]:
        """
        Generates a sequence of tokens using speculative decoding.
        Yields chunks of accepted tokens and the surprise value.
        """
        self.eval()
        tokens = inputs.start_tokens
        total_generated = 0

        # Create a lightweight draft model (can be a smaller version or just a copy)
        draft_model = copy.deepcopy(self)

        while total_generated < inputs.max_new_tokens:
            # 1. Generate a chunk of draft tokens
            draft_tokens = tokens
            with torch.no_grad():
                for _ in range(inputs.speculative_steps):
                    draft_logits, _, _ = draft_model(draft_tokens[:, -self.config.model.max_seq_len:])
                    next_token = self._sample_from_logits(draft_logits[:, -1, :], inputs.temperature, inputs.dynamic_top_k or inputs.top_k)
                    draft_tokens = torch.cat((draft_tokens, next_token), dim=1)

            speculative_chunk = draft_tokens[:, tokens.size(1):]

            # 2. Get true logits and value from the main model
            # Enable grad for surprise calculation
            with torch.enable_grad():
                true_logits, value, _ = self(draft_tokens[:, -self.config.model.max_seq_len:])

            surprise = self._calculate_surprise(value)

            # 3. Validate the draft chunk
            accepted_tokens = []
            for i in range(speculative_chunk.size(1)):
                true_next_token_logits = true_logits[:, i, :]
                draft_token = speculative_chunk[:, i].unsqueeze(-1)

                # Resample from the true model's distribution
                resampled_token = self._sample_from_logits(true_next_token_logits, inputs.temperature, inputs.dynamic_top_k or inputs.top_k)

                if resampled_token.item() == draft_token.item():
                    accepted_tokens.append(draft_token)
                else:
                    # Mismatch found, correct with the resampled token and break
                    accepted_tokens.append(resampled_token)
                    break

            if accepted_tokens:
                accepted_chunk = torch.cat(accepted_tokens, dim=1)
                yield accepted_chunk, surprise
                tokens = torch.cat((tokens, accepted_chunk), dim=1)
                total_generated += accepted_chunk.size(1)
            else:
                # If nothing was accepted (rare), generate one token the normal way to avoid getting stuck
                logits, _, _ = self(tokens[:, -self.config.model.max_seq_len:])
                next_token = self._sample_from_logits(logits[:, -1, :], inputs.temperature, inputs.dynamic_top_k or inputs.top_k)
                yield next_token, surprise
                tokens = torch.cat((tokens, next_token), dim=1)
                total_generated += 1

    @property
    def device(self):
        return self.embedding.embedding.weight.device
