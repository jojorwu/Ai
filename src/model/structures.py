"""
Dataclasses and containers for the Transformer model.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import nn

from src.model.layers.core.embedding import Embedding
from src.model.layers.titans.long_term_memory import LongTermMemory
from src.model.layers.titans.gating import GatingNetwork
from src.model.layers.core.rms_norm import RMSNorm
from src.model.layers.heads.value_head import ValueHead
from src.model.layers.core.vision import VisionEncoder
from src.model.titans.summary import SummaryNetwork


@dataclass
class RopeEmbeddings:
    """
    Dataclass for storing precomputed Rotary Positional Embeddings (RoPE).

    Attributes:
        cos: Precomputed cosine values for RoPE.
        sin: Precomputed sine values for RoPE.
    """
    cos: torch.Tensor
    sin: torch.Tensor


class ModelLayers(nn.Module):
    """
    A container for all layers of the Transformer model.

    This class organizes the different components of the model, such as
    embeddings, decoder blocks, and heads, into a single module.
    """

    def __init__(self, layers: dict[str, nn.Module]) -> None:
        """
        Initializes the ModelLayers container.

        Args:
            layers: A dictionary containing the initialized model layers.
        """
        super().__init__()
        self.embedding: Embedding = layers["embedding"]
        self.long_term_memory: LongTermMemory = layers["long_term_memory"]
        self.gating_network: GatingNetwork = layers["gating_network"]
        self.summary_network: SummaryNetwork = layers["summary_network"]
        self.decoder: nn.ModuleList = layers["decoder"]
        self.post_embedding_norm: RMSNorm = layers["post_embedding_norm"]
        self.final_norm: RMSNorm = layers["final_norm"]
        self.value_head: ValueHead = layers["value_head"]
        self.vision_encoder: VisionEncoder | None = layers.get("vision_encoder")

    def forward(self, *args: Any, **kwargs: Any) -> None:
        """
        This method is not implemented.

        Raises:
            NotImplementedError: ModelLayers does not implement a forward pass.
        """
        raise NotImplementedError(
            "ModelLayers is a container and does not implement a forward pass."
        )


@dataclass
class SamplingConfig:
    """
    Configuration parameters for logit sampling.

    Attributes:
        temperature: Controls randomness in generation.
        top_k: Filters top-k tokens by probability.
        top_p: Filters tokens by cumulative probability (nucleus sampling).
        min_p: Minimum probability threshold relative to the max probability.
        logit_soft_cap: Optional threshold for logit soft-clamping.
        dynamic_top_k: Optional dynamic override for top_k.
    """
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.9
    min_p: float = 0.05
    logit_soft_cap: float | None = None
    dynamic_top_k: int | None = None


@dataclass
class SpeculativeConfig:
    """
    Configuration parameters for speculative decoding.

    Attributes:
        speculative_steps: Number of speculative steps to take.
        value_threshold: Threshold for accepting speculative chunks based on value.
        max_retries: Maximum number of retries for speculative generation.
    """
    speculative_steps: int = 5
    value_threshold: float = -1.0
    max_retries: int = 3


@dataclass
class GenerateInput:
    """
    Input parameters for the generation process.

    Attributes:
        start_tokens: Initial token IDs to start generation from.
        max_new_tokens: Maximum number of new tokens to generate.
        sampling_config: Configuration for sampling.
        speculative_config: Configuration for speculative decoding.
        ltm_override: Optional LTM module to use during generation.
        kv_cache: Optional persistent Key-Value cache.
        draft_cache: Optional persistent Key-Value cache for the draft model.
        ltm_memory: Optional persistent LTM memory matrix.
        stop_tokens: Optional list of token IDs that terminate generation.
        images: Optional image data [batch, channels, height, width].
    """
    start_tokens: torch.Tensor
    max_new_tokens: int
    sampling_config: SamplingConfig = field(default_factory=SamplingConfig)
    speculative_config: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    ltm_override: nn.Module | None = None
    kv_cache: Any | None = None
    draft_cache: Any | None = None
    ltm_memory: torch.Tensor | None = None
    stop_tokens: list[int] | None = None
    images: torch.Tensor | None = None


@dataclass
class ForwardOutput:
    """
    Outputs of the model's forward pass.

    Attributes:
        logits: Logits for the next token prediction.
        value: Predicted value score for the sequence.
        aux_loss: Auxiliary loss (e.g., from MoE).
        ltm_memory: Updated LTM memory matrix.
    """
    logits: torch.Tensor
    value: torch.Tensor
    aux_loss: torch.Tensor
    ltm_memory: torch.Tensor | None = None


@dataclass
class GenerationResult:
    """
    Result of a single generation step.

    Attributes:
        tokens: Generated tokens.
        surprise: Calculated surprise score for the generation step.
        ltm_memory: Updated LTM memory matrix.
    """
    tokens: torch.Tensor
    surprise: float
    ltm_memory: torch.Tensor | None = None
