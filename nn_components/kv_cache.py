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
    def __init__(self, config: KVCacheConfig):
        self.config = config
        cache_shape = (
            config.num_layers,
            config.batch_size,
            config.num_kv_heads,
            config.max_seq_len,
            config.d_k,
        )
        self.k_cache = torch.zeros(cache_shape)
        self.v_cache = torch.zeros(cache_shape)
        self.current_pos = 0

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int):
        """
        Updates the cache with new key and value tensors for a specific layer.
        """
        seq_len = k.shape[2]
        indices = torch.arange(
            self.current_pos, self.current_pos + seq_len
        ) % self.config.max_seq_len
        self.k_cache[layer_idx, :, :, indices, :] = k
        self.v_cache[layer_idx, :, :, indices, :] = v
        if layer_idx == 0:
            self.current_pos += seq_len

    def get(self, layer_idx: int):
        """
        Retrieves the cached keys and values for a specific layer.
        """
        # The actual sequence length in the cache can be up to max_seq_len
        end_pos = self.current_pos
        start_pos = max(0, end_pos - self.config.max_seq_len)
        indices = torch.arange(start_pos, end_pos) % self.config.max_seq_len
        return self.k_cache[layer_idx, ..., indices, :], self.v_cache[layer_idx, ..., indices, :]
