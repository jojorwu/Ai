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
        This allows the model to allocate fewer resources for simpler tokens and more for
        complex ones, optimizing computational efficiency.

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
            - logits: The output logits for next token prediction. Shape: (batch_size, seq_len, vocab_size).
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
        logits = F.linear(
            h, self.layers.embedding.weight
        )  # pylint: disable=not-callable
        # Value head: projects the final hidden state of the last token to a
        # single scalar value, predicting the "usefulness" of the sequence.
        value = self.layers.value_head(h[:, -1, :])
        return logits, value, total_aux_loss

    def _sample_from_logits(self, logits, temperature, top_k, top_p):
        """Samples a token from logits using temperature, top-k, and top-p."""
        if temperature == 0.0:
            _, next_token = torch.topk(logits, k=1, dim=-1)
            return next_token

        logits = logits / temperature

        # Apply top-k
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1, None]] = -float('Inf')

        # Apply top-p (nucleus sampling)
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token


    def _calculate_surprise(
        self, value: torch.Tensor, ltm_override: nn.Module | None = None
    ) -> float:
        """
        Calculates the 'surprise' metric for the LTM update mechanism.

        "Surprise" is a heuristic used to decide when to update the Long-Term Memory.
        It's defined as the norm of the gradients of the LTM's parameters with respect
        to the value head's output. A high surprise value indicates that a small change
        in the LTM would have a large impact on the predicted value, suggesting that the
        LTM's current state is "surprising" or ill-suited for the current context.
        This triggers an LTM update during generation.

        Args:
            value: The scalar tensor output from the value head. It must require gradients.
            ltm_override: The specific LTM module to use (for agent-based training).

        Returns:
            A float representing the calculated surprise value (gradient norm).
        """
        long_term_memory = ltm_override or self.layers.long_term_memory
        if not long_term_memory or not value.requires_grad:
            return 0.0

        # Temporarily calculate gradients for the LTM parameters.
        long_term_memory.zero_grad()
        value.backward(retain_graph=True)
        grad_tensors = [
            p.grad.detach()
            for p in long_term_memory.parameters()
            if p.grad is not None
        ]
        if not grad_tensors:
            return 0.0

        # The surprise is the L2 norm of all LTM gradients concatenated into a single vector.
        surprise = torch.linalg.norm(
            torch.cat([t.flatten() for t in grad_tensors])
        ).item()

        # It's crucial to zero out the gradients afterward so this calculation
        # doesn't interfere with the main training backward pass.
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
                    inputs.sampling_config.dynamic_top_k or inputs.sampling_config.top_k,
                    inputs.sampling_config.top_p,
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

        This is the core of the speculative decoding validation step. It compares the
        tokens proposed by the draft model (`speculative_chunk`) with tokens sampled
        from the main model's more accurate logits (`true_logits`).

        The logic is as follows:
        1. All tokens in the speculative chunk that match the verification tokens are accepted.
        2. If a mismatch occurs at position `i`, all tokens up to `i-1` are accepted.
        3. The correctly sampled token at position `i` from the main model is also accepted.
        4. The rest of the chunk is discarded.
        5. If all tokens match, the entire chunk is accepted.

        This process is fully vectorized for performance.

        Args:
            true_logits: The logits produced by the main model for the speculative sequence.
            speculative_chunk: The chunk of tokens generated by the draft model.
            inputs: The main input object containing sampling configurations.

        Returns:
            A tensor containing the sequence of accepted tokens, which may be shorter
            than the original speculative chunk, or None if validation fails unexpectedly.
        """
        if speculative_chunk.size(0) != 1:
            # This logic is currently only implemented for a batch size of 1.
            raise NotImplementedError(
                "Speculative decoding only supports batch size 1."
            )

        # Sample a sequence of "verification" tokens from the main model's logits.
        # These are what the main model *would have* generated.
        verification_tokens = self._sample_from_logits(
            true_logits.view(-1, true_logits.size(-1)),
            inputs.sampling_config.temperature,
            inputs.sampling_config.dynamic_top_k or inputs.sampling_config.top_k,
            inputs.sampling_config.top_p,
        ).view(speculative_chunk.shape)

        # Find the first point of disagreement between the draft and main models.
        mismatches = (speculative_chunk != verification_tokens).long()

        if not mismatches.any():
            # If there are no mismatches, accept the entire speculative chunk.
            return speculative_chunk

        # Find the index of the first mismatch.
        first_mismatch_idx = torch.argmax(mismatches, dim=1)[0]

        # If the very first token is a mismatch, we can't accept any of the
        # speculative prefix. We just return the first correct token from the
        # main model.
        if first_mismatch_idx == 0 and mismatches[0, 0] == 1:
            return verification_tokens[:, :1]

        # Accept the speculative tokens up to the first mismatch.
        accepted_prefix = speculative_chunk[:, :first_mismatch_idx]
        # Accept the correctly sampled token from the main model at the mismatch point.
        corrected_token = verification_tokens[
            :, first_mismatch_idx : first_mismatch_idx + 1
        ]

        # Concatenate the accepted prefix and the corrected token.
        return torch.cat([accepted_prefix, corrected_token], dim=1)

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
