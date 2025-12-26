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
    Implements a sliding window using a ring buffer.
    """
    def __init__(self, config: KVCacheConfig):
        self.config = config
        cache_shape = (config.num_layers, config.batch_size,
                       config.num_kv_heads, config.max_seq_len, config.d_k)
        self.k_cache = np.zeros(cache_shape)
        self.v_cache = np.zeros(cache_shape)
        self.current_pos = 0
        self.is_filled = False

    def update(self, k, v, layer_idx):
        """
        Updates the cache with new k and v values for the specified layer.
        k, v: (batch_size, num_kv_heads, seq_len, d_k)
        """
        seq_len = k.shape[2]
        start_pos = self.current_pos
        end_pos = start_pos + seq_len

        if end_pos <= self.config.max_seq_len:
            # No wrapping
            self.k_cache[layer_idx, :, :, start_pos:end_pos, :] = k
            self.v_cache[layer_idx, :, :, start_pos:end_pos, :] = v
        else:
            # Handle wrapping around the buffer
            remaining_space = self.config.max_seq_len - start_pos
            # First part
            self.k_cache[layer_idx, :, :, start_pos:, :] = k[:, :, :remaining_space, :]
            self.v_cache[layer_idx, :, :, start_pos:, :] = v[:, :, :remaining_space, :]
            # Wrapped part
            self.k_cache[layer_idx, :, :, :end_pos % self.config.max_seq_len, :] = k[:, :, remaining_space:, :]
            self.v_cache[layer_idx, :, :, :end_pos % self.config.max_seq_len, :] = v[:, :, remaining_space:, :]

        self.current_pos = end_pos % self.config.max_seq_len
        if not self.is_filled and end_pos >= self.config.max_seq_len:
            self.is_filled = True


    def get(self, layer_idx):
        """
        Returns the cached k and v for the specified layer, ordered correctly.
        """
        if not self.is_filled:
            # If buffer is not full, just return up to the current position
            return self.k_cache[layer_idx, ..., :self.current_pos, :], self.v_cache[layer_idx, ..., :self.current_pos, :]

        # If buffer is full (ring buffer logic)
        # The order is from self.current_pos to end, then from start to self.current_pos
        k_rolled = np.roll(self.k_cache[layer_idx], -self.current_pos, axis=-2)
        v_rolled = np.roll(self.v_cache[layer_idx], -self.current_pos, axis=-2)
        return k_rolled, v_rolled


    def snapshot(self):
        """Creates a 'snapshot' of the current state of the cache."""
        return {
            'k_cache': np.copy(self.k_cache),
            'v_cache': np.copy(self.v_cache),
            'current_pos': self.current_pos,
            'is_filled': self.is_filled
        }

    def restore(self, snapshot):
        """Restores the cache state from a 'snapshot'."""
        if 'k_cache' in snapshot and 'v_cache' in snapshot and 'current_pos' in snapshot:
            self.k_cache = np.copy(snapshot['k_cache'])
            self.v_cache = np.copy(snapshot['v_cache'])
            self.current_pos = snapshot['current_pos']
            self.is_filled = snapshot.get('is_filled', False) # For backward compatibility
        else:
            raise ValueError("Invalid snapshot format for KVCache restoration.")

    def copy(self):
        """Creates a deep copy of this KVCache instance."""
        new_cache = KVCache(self.config)
        new_cache.k_cache = np.copy(self.k_cache)
        new_cache.v_cache = np.copy(self.v_cache)
        new_cache.current_pos = self.current_pos
        new_cache.is_filled = self.is_filled
        return new_cache
