"""
PyTorch implementation of the K-V Cache for efficient Transformer generation.
"""
from dataclasses import dataclass

import torch


@dataclass
class KVCacheConfig:
    """Configuration for the KVCache."""
    num_layers: int
    batch_size: int
    num_kv_heads: int
    d_k: int
    max_seq_len: int

class KVCache:
    """
    A Key-Value cache for the Transformer, implemented as a fixed-size ring buffer
    using PyTorch tensors. This is crucial for efficient auto-regressive generation.
    """
    def __init__(self, config: KVCacheConfig, device="cpu", dtype=torch.float32):
        self.config = config
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
        Uses efficient slicing instead of indexing where possible.
        """
        seq_len = k.shape[2]

        # If seq_len > max_seq_len, only take the last max_seq_len tokens
        if seq_len > self.config.max_seq_len:
            k = k[:, :, -self.config.max_seq_len:, :]
            v = v[:, :, -self.config.max_seq_len:, :]
            seq_len = self.config.max_seq_len

        start = self.current_pos % self.config.max_seq_len
        end = start + seq_len

        if end <= self.config.max_seq_len:
            # Normal insertion (no wrap-around within this update)
            self.k_cache[layer_idx, :, :, start:end, :] = k
            self.v_cache[layer_idx, :, :, start:end, :] = v
        else:
            # Wrap-around insertion
            first_part_len = self.config.max_seq_len - start
            self.k_cache[layer_idx, :, :, start:, :] = k[:, :, :first_part_len, :]
            self.v_cache[layer_idx, :, :, start:, :] = v[:, :, :first_part_len, :]

            second_part_len = seq_len - first_part_len
            self.k_cache[layer_idx, :, :, :second_part_len, :] = k[:, :, first_part_len:, :]
            self.v_cache[layer_idx, :, :, :second_part_len, :] = v[:, :, first_part_len:, :]

    def increment_pos(self, seq_len: int):
        """Increments the current position in the cache."""
        self.current_pos += seq_len

    def rollback(self, num_tokens: int):
        """Rolls back the cache position by a certain number of tokens."""
        self.current_pos = max(0, self.current_pos - num_tokens)

    def get(self, layer_idx: int):
        """
        Retrieves the cached keys and values for a specific layer in correct chronological order.
        """
        if self.current_pos == 0:
            return (
                self.k_cache[layer_idx, :, :, :0, :],
                self.v_cache[layer_idx, :, :, :0, :]
            )

        if self.current_pos <= self.config.max_seq_len:
            # Cache is not yet full, no wrap-around needed for retrieval
            return (
                self.k_cache[layer_idx, :, :, :self.current_pos, :],
                self.v_cache[layer_idx, :, :, :self.current_pos, :]
            )

        # Cache is full or has wrapped around.
        # current_pos % max_seq_len is the index of the OLDEST token.
        pos = self.current_pos % self.config.max_seq_len
        if pos == 0:
            # Perfectly aligned
            return self.k_cache[layer_idx], self.v_cache[layer_idx]

        # Re-order the ring buffer to be chronological
        k = torch.cat([
            self.k_cache[layer_idx, :, :, pos:, :],
            self.k_cache[layer_idx, :, :, :pos, :]
        ], dim=2)
        v = torch.cat([
            self.v_cache[layer_idx, :, :, pos:, :],
            self.v_cache[layer_idx, :, :, :pos, :]
        ], dim=2)
        return k, v

    def detach(self):
        """Detaches the cache tensors from the current computation graph."""
        self.k_cache.detach_()
        self.v_cache.detach_()

    def clear(self):
        """Resets the cache."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_pos = 0
