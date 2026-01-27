"""
Dataclasses for the model module.
"""
from dataclasses import dataclass
import torch
from torch import nn

from src.model.layers.embedding import Embedding
from src.model.layers.long_term_memory import LongTermMemory
from src.model.layers.gating import GatingNetwork
from src.model.layers.rms_norm import RMSNorm
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
