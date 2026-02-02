"""
PyTorch implementation of a simple patch-based Vision Encoder.
"""
from __future__ import annotations
import torch
from torch import nn

from src.config.model_config import VisionConfig
from src.model.layers.core.linear import Linear


class VisionEncoder(nn.Module):
    """
    Converts images into a sequence of patch embeddings.

    This is a simplified version of a Vision Transformer (ViT) encoder
    that uses a convolutional layer for patching and a linear projection
    to the model's hidden dimension.
    """

    def __init__(self, config: VisionConfig, d_model: int) -> None:
        """
        Initializes the VisionEncoder.

        Args:
            config: Configuration for the vision encoder.
            d_model: Hidden dimension of the Transformer model.
        """
        super().__init__()
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels

        # Convolutional layer to extract patches and project them
        # in a single step.
        self.patch_proj = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Positional embeddings for patches
        num_patches = (config.image_size[0] // self.patch_size) * (
            config.image_size[1] // self.patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of images into patch embeddings.

        Args:
            images: Input images of shape [batch, channels, height, width].

        Returns:
            Patch embeddings of shape [batch, num_patches, d_model].
        """
        # 1. Patching and projection
        # [batch, d_model, h/p, w/p]
        x = self.patch_proj(images)

        # 2. Flatten patches
        # [batch, d_model, num_patches] -> [batch, num_patches, d_model]
        x = x.flatten(2).transpose(1, 2)

        # 3. Add positional embeddings
        x = x + self.pos_embed

        return x
