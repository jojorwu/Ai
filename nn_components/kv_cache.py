"""
Implementation of the Key-Value (KV) Cache for efficient inference.
"""
from backend import np


class KVCache:
    """
    A Key-Value cache for storing attention states during autoregressive generation.
    """

    def __init__(self, num_layers, batch_size, num_kv_heads, d_k, max_seq_len):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.cache_k = np.zeros((num_layers, batch_size, num_kv_heads, max_seq_len, d_k))
        self.cache_v = np.zeros((num_layers, batch_size, num_kv_heads, max_seq_len, d_k))

    def update(self, k, v, layer_idx, seq_offset):
        """
        Updates the cache with new key and value tensors.
        """
        seq_len = k.shape[2]
        self.cache_k[layer_idx, :, :, seq_offset:seq_offset + seq_len, :] = k
        self.cache_v[layer_idx, :, :, seq_offset:seq_offset + seq_len, :] = v

    def get(self, layer_idx, seq_len):
        """
        Retrieves the cached keys and values up to a given sequence length.
        """
        return self.cache_k[layer_idx, :, :, :seq_len, :], \
            self.cache_v[layer_idx, :, :, :seq_len, :]

    def snapshot(self):
        """
        Returns a copy of the current cache state for speculative decoding.
        """
        return np.copy(self.cache_k), np.copy(self.cache_v)

    def restore(self, state):
        """
        Restores the cache to a previous state from a snapshot.
        """
        self.cache_k, self.cache_v = state

    def copy(self):
        """
        Creates a deep copy of the cache object.
        """
        new_cache = KVCache(self.num_layers, self.batch_size, self.num_kv_heads,
                            self.d_k, self.max_seq_len)
        new_cache.cache_k = np.copy(self.cache_k)
        new_cache.cache_v = np.copy(self.cache_v)
        return new_cache
