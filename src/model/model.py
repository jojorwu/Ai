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
from bitsandbytes.nn import Linear4bit

from src.config import DecoderBlockConfig, TransformerConfig
from src.model.layers.decoder_block import DecoderBlock, ForwardPassInput
from src.model.layers.embedding import Embedding
from src.model.layers.gating import GatingNetwork
from src.model.layers.linear import Linear
from src.model.layers.long_term_memory import LongTermMemory
from src.model.layers.rms_norm import RMSNorm
from src.model.layers.rotary_embedding import precompute_rope_embeddings
from src.model.layers.value_head import ValueHead


@dataclass
class RopeEmbeddings:
    """Dataclass for storing rope embeddings."""
    cos: torch.Tensor
    sin: torch.Tensor


class ModelLayers(nn.Module):
    """Container for model layers."""

    def __init__(self, layers: dict[str, nn.Module]):
        super().__init__()
        self.embedding: Embedding = layers["embedding"]
        self.long_term_memory: LongTermMemory = layers["long_term_memory"]
        self.gating_network: GatingNetwork = layers["gating_network"]
        self.decoder: nn.ModuleList = layers["decoder"]
        self.final_norm: RMSNorm = layers["final_norm"]
        self.value_head: ValueHead = layers["value_head"]

    def forward(self, *args, **kwargs):
        """This method is not implemented."""
        raise NotImplementedError(
            "ModelLayers is a container and does not implement a forward pass."
        )


@dataclass
class SamplingConfig:
    """Configuration for sampling."""
    temperature: float = 1.0
    top_k: int = 0
    dynamic_top_k: int = None


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    speculative_steps: int = 5
    value_threshold: float = -1.0
    max_retries: int = 3


@dataclass
class GenerateInput:
    """Datacaclass for storing inputs to the generate method."""

    start_tokens: torch.Tensor
    max_new_tokens: int
    sampling_config: SamplingConfig = field(default_factory=SamplingConfig)
    speculative_config: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    ltm_override: nn.Module | None = None


class Transformer(nn.Module):
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
        self.rope_embeddings = self._init_rope_embeddings(config)
        self.layers = self._init_layers(config, load_in_4bit)
        self.layers.embedding.weight = self.layers.embedding.embedding.weight
        self.draft_model = self._create_draft_model(config, load_in_4bit)

    def _create_draft_model(
        self, config: TransformerConfig, load_in_4bit: bool
    ) -> "Transformer|None":
        """
        Creates a smaller, faster 'draft' model for speculative decoding.
        This model is created once during initialization to avoid the expensive
        `copy.deepcopy()` operation during generation.

        The draft model has fewer layers, making it faster but less accurate.
        If the base model is already too small, no draft model is created.
        """
        if config.model.num_layers < 2:
            return None # Don't create a draft model for very small models.

        draft_config_dict = config.model_dump()
        # Reduce the number of layers for the draft model, e.g., by half.
        draft_config_dict["model"]["num_layers"] //= 2

        draft_config = TransformerConfig(**draft_config_dict)

        logging.info(
            "Creating a draft model with %d layers.",
            draft_config.model.num_layers,
        )
        return Transformer(draft_config, load_in_4bit)

    def count_parameters(self):
        """Counts the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_layers(self, config: TransformerConfig, load_in_4bit: bool) -> ModelLayers:
        """Initializes all layers of the model."""
        ltm = self._init_ltm(config)
        decoder_config = self._create_block_config(config, ltm, load_in_4bit)
        value_head_linear_class = Linear4bit if load_in_4bit else Linear

        embedding = Embedding(config.vocab_size, config.model.d_model)
        gating_network = GatingNetwork(
            d_model=config.model.d_model,
            num_layers=config.model.num_layers,
            num_experts=config.model.num_experts,
        )
        decoder = nn.ModuleList(
            [DecoderBlock(decoder_config) for _ in range(config.model.num_layers)]
        )
        final_norm = RMSNorm(config.model.d_model)
        value_head = ValueHead(
            config.model.d_model, linear_class=value_head_linear_class
        )

        layers_dict = {
            "embedding": embedding,
            "long_term_memory": ltm,
            "gating_network": gating_network,
            "decoder": decoder,
            "final_norm": final_norm,
            "value_head": value_head,
        }
        return ModelLayers(layers_dict)

    def _init_ltm(self, config: TransformerConfig) -> LongTermMemory | None:
        """Initializes the Long-Term Memory (LTM) module if configured."""
        if config.model.ltm.d_hidden and config.model.ltm.num_layers:
            return LongTermMemory(
                d_model=config.model.d_model,
                d_hidden=config.model.ltm.d_hidden,
                num_layers=config.model.ltm.num_layers,
            )
        return None

    def _init_rope_embeddings(self, config: TransformerConfig) -> RopeEmbeddings:
        """Initializes and registers Rotary Positional Embeddings (RoPE)."""
        d_k = config.model.d_model // config.model.num_heads
        rope_cos, rope_sin = precompute_rope_embeddings(
            d_k, config.model.max_seq_len
        )
        self.register_buffer("rope_cos_buf", rope_cos)
        self.register_buffer("rope_sin_buf", rope_sin)
        return RopeEmbeddings(cos=self.rope_cos_buf, sin=self.rope_sin_buf)

    def _create_block_config(
        self, config: TransformerConfig, ltm: nn.Module, load_in_4bit: bool
    ) -> DecoderBlockConfig:
        """Helper method to create the DecoderBlockConfig."""
        return DecoderBlockConfig(
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            d_ff=config.model.d_ff,
            dropout_rate=config.model.dropout_rate,
            num_kv_heads=config.model.num_kv_heads,
            rotary_emb=(self.rope_embeddings.cos, self.rope_embeddings.sin),
            num_layers=config.model.num_layers,
            long_term_memory=ltm,
            num_experts=config.model.num_experts,
            top_k_experts=config.model.top_k_experts,
            load_in_4bit=load_in_4bit,
        )

    def forward( # pylint: disable=too-many-locals
        self,
        x: torch.Tensor,
        ltm_state: torch.Tensor = None,
        dynamic_top_k: int = None,  # This is passed in from generation config
        ltm_override: nn.Module | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the Transformer model.

        This method includes the logic for dynamic architecture adjustments,
        such as layer skipping and adaptive MoE, based on the LTM's complexity score.

        Args:
            x: Input tensor of token IDs.
            ltm_state: An optional pre-computed state from the Long-Term Memory.
                       If None, it will be computed from the input `x`.
            dynamic_top_k: An optional integer to override the dynamic MoE k-value.
            ltm_override: An optional LTM module to use instead of the model's default.
                          This is used for agent-specific LTMs.

        Returns:
            A tuple containing:
            - logits: The output logits for the next token prediction.
            - value: The predicted value from the value head.
            - total_aux_loss: The auxiliary loss from the MoE layers.
        """
        # 1. Get token embeddings
        h = self.layers.embedding(x) * math.sqrt(self.config.model.d_model)

        # 2. Determine and compute the Long-Term Memory state
        long_term_memory = ltm_override or self.layers.long_term_memory
        if ltm_state is None:
            if long_term_memory:
                # If no LTM state is provided, compute it from the input sequence
                ltm_state, _ = long_term_memory(
                    h.mean(dim=1, keepdim=True) # Use mean of sequence as summary
                )
            else:
                # If there's no LTM, use a zero tensor as a placeholder
                ltm_state = torch.zeros(
                    (h.size(0), 1, h.size(2)), device=h.device, dtype=h.dtype
                )

        # 3. Get dynamic parameters from the GatingNetwork
        active_layers_tensor, moe_top_k = self.layers.gating_network(ltm_state)
        # For the decoder loop, we need a single integer. Since generation has a
        # batch size of 1, taking the max works for both cases.
        active_layers = int(torch.max(active_layers_tensor).item())

        # Generation config can override the LTM's dynamic selection
        final_dynamic_top_k_tensor = dynamic_top_k if dynamic_top_k is not None else moe_top_k
        # For the MoE layer, we need a single integer.
        final_dynamic_top_k = int(torch.max(final_dynamic_top_k_tensor).item())


        # 4. Pass through the dynamically selected number of decoder blocks

        total_aux_loss = torch.tensor(0.0, device=x.device)
        for i in range(active_layers):
            block = self.layers.decoder[i]
            inputs = ForwardPassInput(
                x=h,
                ltm_state=ltm_state,
                layer_idx=i,
                dynamic_top_k=final_dynamic_top_k,
            )
            h, aux_loss = block(inputs)
            if aux_loss is not None:
                total_aux_loss += aux_loss

        h = self.layers.final_norm(h)
        logits = F.linear(h, self.layers.embedding.weight) # pylint: disable=not-callable
        value = self.layers.value_head(h[:, -1, :])
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


    def _calculate_surprise(
        self, value: torch.Tensor, ltm_override: nn.Module | None = None
    ) -> float:
        """Calculates 'surprise' by backpropagating value and getting LTM grad norm."""
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

        surprise = torch.norm(torch.cat([t.flatten() for t in grad_tensors])).item()
        long_term_memory.zero_grad()
        return surprise

    def _generate_speculative_chunk(
        self, draft_model: "Transformer", tokens: torch.Tensor, inputs: GenerateInput
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a small 'draft' chunk of tokens using a faster, smaller model.

        Args:
            draft_model: The smaller, faster model used for generating speculative tokens.
            tokens: The current sequence of tokens generated so far.
            inputs: The main input object containing generation configurations.

        Returns:
            A tuple containing:
            - The newly generated speculative chunk.
            - The full sequence including the speculative chunk.
        """
        draft_tokens = tokens
        with torch.no_grad():
            for _ in range(inputs.speculative_config.speculative_steps):
                # Generate one token at a time with the draft model
                draft_logits, _, _ = draft_model(
                    draft_tokens[:, -self.config.model.max_seq_len :]
                )
                next_token = self._sample_from_logits(
                    draft_logits[:, -1, :],
                    inputs.sampling_config.temperature,
                    inputs.sampling_config.dynamic_top_k
                    or inputs.sampling_config.top_k,
                )
                draft_tokens = torch.cat((draft_tokens, next_token), dim=1)
        # Return only the newly generated tokens and the full draft sequence
        return draft_tokens[:, tokens.size(1) :], draft_tokens

    def _validate_and_accept_chunk(
        self,
        true_logits: torch.Tensor,
        speculative_chunk: torch.Tensor,
        inputs: GenerateInput,
    ) -> torch.Tensor | None:
        """
        Validates a speculative chunk against logits from the main model.

        It compares the tokens proposed by the draft model with tokens sampled from
        the main model's logits. Tokens are accepted up to the first mismatch.

        Args:
            true_logits: The logits produced by the main model for the speculative sequence.
            speculative_chunk: The chunk of tokens generated by the draft model.
            inputs: The main input object containing sampling configurations.

        Returns:
            A tensor containing the sequence of accepted tokens, which may be shorter
            than the original speculative chunk.
        """
        if speculative_chunk.size(0) != 1:
            raise NotImplementedError(
                "Speculative decoding only supports batch size 1."
            )

        # Sample verification tokens from the main model's logits
        verification_tokens = self._sample_from_logits(
            true_logits.view(-1, true_logits.size(-1)),
            inputs.sampling_config.temperature,
            inputs.sampling_config.dynamic_top_k or inputs.sampling_config.top_k,
        ).view(speculative_chunk.shape)

        accepted_tokens = []
        for i in range(speculative_chunk.size(1)):
            draft_token = speculative_chunk[0, i]
            ver_token = verification_tokens[0, i]
            if draft_token == ver_token:
                accepted_tokens.append(draft_token.unsqueeze(0))
            else:
                accepted_tokens.append(ver_token.unsqueeze(0))
                break

        return torch.cat(accepted_tokens, dim=0).unsqueeze(0) if accepted_tokens else None

    def generate(self, inputs: GenerateInput) -> Generator[Tuple[torch.Tensor, float], None, None]:
        """
        Generates a sequence of tokens using speculative decoding.
        Yields chunks of accepted tokens and the surprise value.
        """
        self.eval()
        # Use the pre-initialized draft model, or default to self if it's not available.
        draft_model = self.draft_model or self
        tokens = inputs.start_tokens.to(self.device)
        total_generated = 0

        while total_generated < inputs.max_new_tokens:
            speculative_chunk, draft_tokens = self._generate_speculative_chunk(
                draft_model, tokens, inputs
            )

            with torch.enable_grad():
                true_logits, value, _ = self(
                    draft_tokens[:, -self.config.model.max_seq_len :],
                    ltm_override=inputs.ltm_override,
                )
            surprise = self._calculate_surprise(value, inputs.ltm_override)

            accepted_chunk = self._validate_and_accept_chunk(
                true_logits, speculative_chunk, inputs
            )

            if accepted_chunk is not None:
                yield accepted_chunk, surprise
                tokens = torch.cat((tokens, accepted_chunk), dim=1)
                total_generated += accepted_chunk.size(1)
            else:
                logits, _, _ = self(tokens[:, -self.config.model.max_seq_len :])
                next_token = self._sample_from_logits(
                    logits[:, -1, :],
                    inputs.sampling_config.temperature,
                    inputs.sampling_config.dynamic_top_k
                    or inputs.sampling_config.top_k,
                )
                yield next_token, surprise
                tokens = torch.cat((tokens, next_token), dim=1)
                total_generated += 1

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
