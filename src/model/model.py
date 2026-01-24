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

from src.config.model_config import DecoderBlockConfig, TransformerConfig
from src.model.layers.decoder_block import DecoderBlock, ForwardPassInput
from src.model.layers.embedding import Embedding
from src.model.layers.gating import GatingNetwork
from src.model.layers.linear import Linear
from src.model.layers.long_term_memory import LongTermMemory
from src.model.layers.rms_norm import RMSNorm
from src.model.layers.rotary_embedding import precompute_rope_embeddings
from src.model.layers.value_head import ValueHead
from src.model.generation import GenerationMixin


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
    top_p: float = 0.9
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

        self.rope_embeddings = self._init_rope_embeddings()
        self.layers = self._init_layers()
        self.draft_model = self._init_draft_model()

        # Weight tying: share weights between embedding and policy head
        self.layers.embedding.weight = self.layers.embedding.embedding.weight

    def _init_rope_embeddings(self) -> RopeEmbeddings:
        """Initializes and registers Rotary Positional Embeddings (RoPE)."""
        d_k = self.config.model.d_model // self.config.model.num_heads
        rope_cos, rope_sin = precompute_rope_embeddings(
            d_k, self.config.model.max_seq_len
        )
        self.register_buffer("rope_cos_buf", rope_cos.clone())
        self.register_buffer("rope_sin_buf", rope_sin.clone())
        return RopeEmbeddings(cos=self.rope_cos_buf, sin=self.rope_sin_buf)

    def _init_layers(self) -> ModelLayers:
        """Initializes all layers of the model."""
        ltm = self._init_ltm()
        decoder_config = self._create_decoder_block_config(ltm)
        value_head_linear_class = Linear4bit if self.load_in_4bit else Linear

        layers_dict = {
            "embedding": Embedding(self.config.vocab_size, self.config.model.d_model),
            "long_term_memory": ltm,
            "gating_network": GatingNetwork(
                d_model=self.config.model.d_model,
                num_layers=self.config.model.num_layers,
                num_experts=self.config.model.num_experts,
            ),
            "decoder": nn.ModuleList(
                [
                    DecoderBlock(decoder_config)
                    for _ in range(self.config.model.num_layers)
                ]
            ),
            "final_norm": RMSNorm(self.config.model.d_model),
            "value_head": ValueHead(
                self.config.model.d_model, linear_class=value_head_linear_class
            ),
        }
        return ModelLayers(layers_dict)

    def _init_ltm(self) -> LongTermMemory | None:
        """Initializes the Long-Term Memory (LTM) module if configured."""
        cfg = self.config.model.ltm
        if cfg.d_hidden and cfg.num_layers:
            return LongTermMemory(
                d_model=self.config.model.d_model,
                d_hidden=cfg.d_hidden,
                num_layers=cfg.num_layers,
            )
        return None

    def _create_decoder_block_config(self, ltm: nn.Module) -> DecoderBlockConfig:
        """Helper method to create the DecoderBlockConfig."""
        model_cfg = self.config.model
        return DecoderBlockConfig(
            d_model=model_cfg.d_model,
            num_heads=model_cfg.num_heads,
            d_ff=model_cfg.d_ff,
            dropout_rate=model_cfg.dropout_rate,
            num_kv_heads=model_cfg.num_kv_heads,
            rotary_emb=(self.rope_embeddings.cos, self.rope_embeddings.sin),
            num_layers=model_cfg.num_layers,
            long_term_memory=ltm,
            num_experts=model_cfg.num_experts,
            top_k_experts=model_cfg.top_k_experts,
            load_in_4bit=self.load_in_4bit,
        )

    def _init_draft_model(self) -> "Transformer|None":
        """
        Creates a smaller, faster 'draft' model for speculative decoding.
        """
        if self.config.model.num_layers < 2:
            return None

        draft_config_dict = self._config.model_dump()
        draft_config_dict["model"]["num_layers"] //= 2
        draft_config = TransformerConfig(**draft_config_dict)

        logging.info(
            "Creating a draft model with %d layers.",
            draft_config.model.num_layers,
        )
        return Transformer(draft_config, self.load_in_4bit)

    def count_parameters(self):
        """Counts the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(  # pylint: disable=too-many-locals
        self,
        x: torch.Tensor,
        ltm_state: torch.Tensor = None,
        dynamic_top_k: int = None,  # This is passed in from generation config
        ltm_override: nn.Module | None = None,
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
        final_dynamic_top_k_tensor = (
            dynamic_top_k if dynamic_top_k is not None else moe_top_k
        )
        # For the MoE layer, we also need a single integer.
        final_dynamic_top_k = int(torch.max(final_dynamic_top_k_tensor).item())

        # 4. Pass through the dynamically selected number of decoder blocks.
        # The model only computes the number of layers determined by the GatingNetwork,
        # saving significant computation on simpler tokens.
        total_aux_loss = torch.tensor(0.0, device=x.device)
        for i in range(active_layers):
            block = self.layers.decoder[i]
            block_input = ForwardPassInput(
                x=h,
                ltm_state=ltm_state,
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
        return logits, value, total_aux_loss

    def generate(
        self, inputs: GenerateInput
    ) -> Generator[Tuple[torch.Tensor, float], None, None]:
        """
        Generates a sequence of tokens using speculative decoding.

        This method accelerates generation by using a smaller, faster 'draft' model
        to produce a chunk of speculative tokens. The main, more powerful model then
        validates this entire chunk in a single, parallel forward pass. Tokens from the
        chunk are accepted up to the first point of disagreement, and then the process
        repeats. This is significantly faster than standard auto-regressive generation,
        which requires one full forward pass for every single token.

        This method is a generator, yielding `(accepted_chunk, surprise_value)` tuples
        as they are produced, allowing for incremental streaming of the output.

        Args:
            inputs: A dataclass containing all necessary parameters for generation,
                    including start tokens, max length, and sampling settings.

        Yields:
            A tuple containing:
            - A tensor of the accepted token chunk (can be one or more tokens).
            - A float representing the 'surprise' value calculated during this step.
        """
        self.eval()
        # Use the pre-initialized draft model. If it's not available (e.g., for
        # very small models where the overhead isn't worth it), default to the main
        # model itself, effectively falling back to auto-regressive sampling.
        draft_model = self.draft_model or self
        tokens = inputs.start_tokens.to(self.device)
        total_generated = 0

        while total_generated < inputs.max_new_tokens:
            # 1. Generate a speculative chunk using the smaller, faster draft model.
            speculative_chunk, draft_tokens = self._generate_speculative_chunk(
                draft_model, tokens, inputs
            )

            # 2. Run a single forward pass on the main model to get the true logits
            # for the entire speculative sequence. Gradients are enabled to allow for
            # the 'surprise' calculation for the LTM.
            with torch.enable_grad():
                true_logits, value, _ = self(
                    draft_tokens[:, -self.config.model.max_seq_len :],
                    ltm_override=inputs.ltm_override,
                )

            # 3. Calculate the 'surprise' metric to determine if the LTM should be updated.
            surprise = self._calculate_surprise(value, inputs.ltm_override)

            # 4. Validate the speculative chunk. We only need the logits corresponding
            # to the newly generated tokens for this.
            validation_logits = true_logits[:, -speculative_chunk.size(1) - 1 : -1, :]
            accepted_chunk = self._validate_and_accept_chunk(
                validation_logits, speculative_chunk, inputs
            )

            # 5. Yield the accepted tokens and update the main sequence.
            if accepted_chunk is not None:
                yield accepted_chunk, surprise
                tokens = torch.cat((tokens, accepted_chunk), dim=1)
                total_generated += accepted_chunk.size(1)
            else:
                # This 'else' block is a fallback and should rarely be hit with the
                # current validation logic. It handles the case where validation
                # might fail entirely by reverting to standard auto-regressive sampling
                # for one token to ensure progress is always made.
                logits, _, _ = self(tokens[:, -self.config.model.max_seq_len :])
                next_token = self._sample_from_logits(
                    logits[:, -1, :],
                    inputs.sampling_config.temperature,
                    inputs.sampling_config.dynamic_top_k
                    or inputs.sampling_config.top_k,
                    inputs.sampling_config.top_p,
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
