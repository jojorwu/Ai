"""
Dataclasses for the model module.
"""
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import nn

from src.model.layers.core.embedding import Embedding
from src.model.layers.titans.long_term_memory import LongTermMemory
from src.model.layers.titans.gating import GatingNetwork
from src.model.layers.core.rms_norm import RMSNorm
from src.model.layers.heads.value_head import ValueHead


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
    min_p: float = 0.05
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
    kv_cache: Any = None
    draft_cache: Any = None
    ltm_memory: Optional[torch.Tensor] = None
