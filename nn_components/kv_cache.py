"""
This module implements the Key-Value Cache for the self-attention layers.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class KVCacheConfig:
    """Configuration for the KV cache."""
    num_layers: int
    batch_size: int
    num_kv_heads: int
    d_k: int
    max_seq_len: int


class KVCache:
    """
    Cache for key-value pairs in self-attention layers, optimized for GQA.
    """
    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.k_cache = np.zeros((config.num_layers, config.batch_size, config.num_kv_heads, config.max_seq_len, config.d_k))
        self.v_cache = np.zeros((config.num_layers, config.batch_size, config.num_kv_heads, config.max_seq_len, config.d_k))

    def update(self, k, v, layer_idx, seq_offset):
        """
        Updates the cache with new k and v values for the specified layer.
        k, v: (batch_size, num_kv_heads, seq_len, d_k)
        seq_offset: the starting position for insertion
        """
        seq_len = k.shape[2]
        end_pos = seq_offset + seq_len

        if end_pos > self.config.max_seq_len:
            raise ValueError("KVCache: sequence exceeds maximum length.")

        self.k_cache[layer_idx, :, :, seq_offset:end_pos, :] = k
        self.v_cache[layer_idx, :, :, seq_offset:end_pos, :] = v

    def get(self, layer_idx, seq_len):
        """
        Returns the cached k and v for the specified layer up to a certain length.
        """
        return self.k_cache[layer_idx, ..., :seq_len, :], self.v_cache[layer_idx, ..., :seq_len, :]

    def snapshot(self):
        """Creates a 'snapshot' of the current state of the cache."""
        return {'k_cache': np.copy(self.k_cache), 'v_cache': np.copy(self.v_cache)}

    def restore(self, snapshot):
        """Restores the cache state from a 'snapshot'."""
        if 'k_cache' in snapshot and 'v_cache' in snapshot:
            self.k_cache = np.copy(snapshot['k_cache'])
            self.v_cache = np.copy(snapshot['v_cache'])
        else:
            raise ValueError("Invalid snapshot format for KVCache restoration.")

    def copy(self):
        """Creates a deep copy of this KVCache instance."""
        new_cache = KVCache(self.config)
        new_cache.k_cache = np.copy(self.k_cache)
        new_cache.v_cache = np.copy(self.v_cache)
        return new_cache
