"""
This module implements the Key-Value Cache for the self-attention layers.
"""
import numpy as np


class KVCache:
    """
    Cache for key-value pairs in self-attention layers, optimized for GQA.
    """
    def __init__(self, num_layers, batch_size, num_kv_heads, d_k, max_seq_len):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        self.k_cache = np.zeros((num_layers, batch_size, num_kv_heads, max_seq_len, d_k))
        self.v_cache = np.zeros((num_layers, batch_size, num_kv_heads, max_seq_len, d_k))

    def update(self, k, v, layer_idx, seq_offset):
        """
        Updates the cache with new k and v values for the specified layer.
        k, v: (batch_size, num_kv_heads, seq_len, d_k)
        seq_offset: the starting position for insertion
        """
        seq_len = k.shape[2]
        end_pos = seq_offset + seq_len

        if end_pos > self.max_seq_len:
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
        new_cache = KVCache(self.num_layers, self.batch_size, self.num_kv_heads, self.d_k, self.max_seq_len)
        new_cache.k_cache = np.copy(self.k_cache)
        new_cache.v_cache = np.copy(self.v_cache)
        return new_cache
