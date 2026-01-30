"""
PyTorch implementation of the K-V Cache for efficient Transformer generation.
"""
from dataclasses import dataclass
import logging

import torch


@dataclass
class KVCacheConfig:
    """Configuration for the KVCache."""
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
    def __init__(self, config: KVCacheConfig, device="cpu", dtype=torch.float32):
        self.config = config
        # Ensure anchor_size is reasonable
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

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int):
        """
        Updates the cache with new key and value tensors for a specific layer.
        Preserves anchors and uses a ring buffer for the rest.
        """
        if k.numel() == 0:
            return

        seq_len = k.shape[2]
        anchor_size = self.config.anchor_size
        max_seq_len = self.config.max_seq_len
        sliding_capacity = max_seq_len - anchor_size

        # Safety check for sliding capacity
        if sliding_capacity <= 0:
            # If no sliding window is possible, we just fill up to max_seq_len and stop updating
            # or treat everything as an anchor. Here we treat everything as an anchor.
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

    def increment_pos(self, seq_len: int):
        """Increments the current position in the cache."""
        self.current_pos += seq_len

    def rollback(self, num_tokens: int):
        """Rolls back the cache position by a certain number of tokens."""
        self.current_pos = max(0, self.current_pos - num_tokens)

    def get(self, layer_idx: int, seq_len: int = 0):
        """
        Retrieves the cached keys and values for a specific layer in correct chronological order.
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

    def detach(self):
        """Detaches the cache tensors from the current computation graph."""
        self.k_cache.detach_()
        self.v_cache.detach_()

    def clear(self):
        """Resets the cache."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_pos = 0
