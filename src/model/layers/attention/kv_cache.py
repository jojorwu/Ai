"""
PyTorch implementation of the K-V Cache for efficient Transformer generation.
"""
from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Tuple

import torch


@dataclass
class KVCacheConfig:
    """
    Configuration for the KVCache.

    Attributes:
        num_layers: Number of layers in the model.
        batch_size: Batch size for inference.
        num_kv_heads: Number of Key-Value heads.
        d_k: Dimension of each head.
        max_seq_len: Maximum sequence length supported by the cache.
        anchor_size: Number of initial tokens to keep as fixed anchors (Attention Sinks).
    """
    num_layers: int
    batch_size: int
    num_kv_heads: int
    d_k: int
    max_seq_len: int
    anchor_size: int = 4


class KVCache:
    """
    A Key-Value cache for the Transformer, implemented as a fixed-size ring buffer
    with support for fixed anchors (Attention Sinks).
    """

    def __init__(
        self, config: KVCacheConfig, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32
    ) -> None:
        """
        Initializes the KVCache.

        Args:
            config: Configuration object for the cache.
            device: Device to allocate the cache on.
            dtype: Data type for the cache tensors.
        """
        self.config = config

        # Harden anchor_size validation
        if not isinstance(self.config.anchor_size, int) or self.config.anchor_size < 0:
            logging.warning("anchor_size must be a non-negative integer. Resetting to 0.")
            self.config.anchor_size = 0

        if self.config.anchor_size >= self.config.max_seq_len:
            logging.warning("anchor_size >= max_seq_len. Disabling anchors for stability.")
            self.config.anchor_size = 0

        cache_shape = (
            config.num_layers,
            config.batch_size,
            config.num_kv_heads,
            config.max_seq_len,
            config.d_k,
        )
        self.k_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.current_pos = 0

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int) -> None:
        """
        Updates the cache with new key and value tensors for a specific layer.

        Args:
            k: New key tensor of shape [batch, heads, seq_len, d_k].
            v: New value tensor of shape [batch, heads, seq_len, d_k].
            layer_idx: Index of the layer being updated.
        """
        if k.numel() == 0:
            return

        seq_len = k.shape[2]
        anchor_size = self.config.anchor_size
        max_seq_len = self.config.max_seq_len
        sliding_capacity = max_seq_len - anchor_size

        # Safety check for sliding capacity
        if sliding_capacity <= 0:
            fill_len = min(seq_len, max_seq_len - self.current_pos)
            if fill_len > 0:
                indices = torch.arange(fill_len, device=k.device) + self.current_pos
                self.k_cache[layer_idx, :, :, indices, :] = k[:, :, :fill_len, :]
                self.v_cache[layer_idx, :, :, indices, :] = v[:, :, :fill_len, :]
            return

        # For very large prompts, we can only keep anchor_size + sliding_capacity tokens.
        if seq_len > max_seq_len:
            k_anchors = k[:, :, :anchor_size, :]
            v_anchors = v[:, :, :anchor_size, :]
            k_sliding = k[:, :, -sliding_capacity:, :]
            v_sliding = v[:, :, -sliding_capacity:, :]

            self.k_cache[layer_idx, :, :, :anchor_size, :] = k_anchors
            self.v_cache[layer_idx, :, :, :anchor_size, :] = v_anchors
            self.k_cache[layer_idx, :, :, anchor_size:, :] = k_sliding
            self.v_cache[layer_idx, :, :, anchor_size:, :] = v_sliding
            return

        indices = torch.arange(seq_len, device=k.device) + self.current_pos
        is_anchor = indices < anchor_size

        # Ring buffer logic for non-anchor tokens
        sliding_indices = anchor_size + (indices - anchor_size) % sliding_capacity
        cache_indices = torch.where(is_anchor, indices, sliding_indices)

        # Stability: Ensure indices are within valid range [0, max_seq_len-1]
        cache_indices = torch.clamp(cache_indices, 0, max_seq_len - 1)

        # Vectorized assignment to the cache
        self.k_cache[layer_idx, :, :, cache_indices, :] = k
        self.v_cache[layer_idx, :, :, cache_indices, :] = v

    def increment_pos(self, seq_len: int) -> None:
        """
        Increments the current position in the cache.

        Args:
            seq_len: Number of tokens added to the cache.
        """
        self.current_pos += seq_len

    def rollback(self, num_tokens: int) -> None:
        """
        Rolls back the cache position.

        Args:
            num_tokens: Number of tokens to roll back.
        """
        self.current_pos = max(0, self.current_pos - num_tokens)

    def get(self, layer_idx: int, seq_len: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the cached keys and values for a specific layer.

        Args:
            layer_idx: Index of the layer to retrieve.
            seq_len: Additional sequence length to include in retrieval.

        Returns:
            A tuple of (key_cache, value_cache) tensors.
        """
        effective_pos = self.current_pos + seq_len
        anchor_size = self.config.anchor_size
        max_seq_len = self.config.max_seq_len
        sliding_capacity = max_seq_len - anchor_size

        if effective_pos == 0:
            return (
                self.k_cache[layer_idx, :, :, :0, :],
                self.v_cache[layer_idx, :, :, :0, :]
            )

        if effective_pos <= max_seq_len:
            # No wrap-around yet
            return (
                self.k_cache[layer_idx, :, :, :effective_pos, :],
                self.v_cache[layer_idx, :, :, :effective_pos, :]
            )

        # If sliding window is disabled (capacity <= 0), we just return the full cache.
        if sliding_capacity <= 0:
            return self.k_cache[layer_idx], self.v_cache[layer_idx]

        # Fast-path: if anchor_size is 0 and pos_in_sliding is 0, return everything
        pos_in_sliding = (effective_pos - anchor_size) % sliding_capacity
        if anchor_size == 0 and pos_in_sliding == 0:
            return self.k_cache[layer_idx], self.v_cache[layer_idx]

        # Anchors are always the first anchor_size tokens
        k_anchors = self.k_cache[layer_idx, :, :, :anchor_size, :]
        v_anchors = self.v_cache[layer_idx, :, :, :anchor_size, :]

        # Re-order sliding part to maintain chronological consistency
        k_sliding_part = self.k_cache[layer_idx, :, :, anchor_size:, :]
        v_sliding_part = self.v_cache[layer_idx, :, :, anchor_size:, :]

        if pos_in_sliding == 0:
            k_sliding, v_sliding = k_sliding_part, v_sliding_part
        else:
            k_sliding = torch.cat([
                k_sliding_part[:, :, pos_in_sliding:, :],
                k_sliding_part[:, :, :pos_in_sliding, :]
            ], dim=2)
            v_sliding = torch.cat([
                v_sliding_part[:, :, pos_in_sliding:, :],
                v_sliding_part[:, :, :pos_in_sliding, :]
            ], dim=2)

        if anchor_size == 0:
            return k_sliding, v_sliding

        return (
            torch.cat([k_anchors, k_sliding], dim=2),
            torch.cat([v_anchors, v_sliding], dim=2)
        )

    def detach(self) -> None:
        """Detaches the cache tensors from the current computation graph."""
        self.k_cache.detach_()
        self.v_cache.detach_()

    def clear(self) -> None:
        """Resets the cache position and clears the stored tensors."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_pos = 0
