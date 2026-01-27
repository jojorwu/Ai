"""
Tests for the PyTorch-based KVCache.
"""
import unittest

import torch

from src.model.layers.kv_cache import KVCache, KVCacheConfig


class TestKVCache(unittest.TestCase):
    """
    Tests for the PyTorch KVCache.
    """

    def setUp(self):
        """Set up a KVCache instance for testing."""
        self.config = KVCacheConfig(
            num_layers=2,
            batch_size=4,
            num_kv_heads=8,
            d_k=64,
            max_seq_len=10
        )
        self.kv_cache = KVCache(self.config)

    def test_initialization(self):
        """Tests that the cache is initialized with the correct shapes."""
        expected_shape = (
            self.config.num_layers,
            self.config.batch_size,
            self.config.num_kv_heads,
            self.config.max_seq_len,
            self.config.d_k,
        )
        self.assertEqual(self.kv_cache.k_cache.shape, expected_shape)
        self.assertEqual(self.kv_cache.v_cache.shape, expected_shape)
        self.assertEqual(self.kv_cache.current_pos, 0)

    def test_update_and_get(self):
        """Tests the update and get methods."""
        layer_idx, seq_len = 0, 5
        k_new = torch.randn(
            self.config.batch_size, self.config.num_kv_heads, seq_len, self.config.d_k
        )
        v_new = torch.randn(
            self.config.batch_size, self.config.num_kv_heads, seq_len, self.config.d_k
        )
        self.kv_cache.update(k_new, v_new, layer_idx)
        self.kv_cache.increment_pos(seq_len)
        self.assertEqual(self.kv_cache.current_pos, seq_len)
        k_cached, v_cached = self.kv_cache.get(layer_idx)
        expected_shape = (
            self.config.batch_size,
            self.config.num_kv_heads,
            seq_len,
            self.config.d_k,
        )
        self.assertEqual(k_cached.shape, expected_shape)
        self.assertEqual(v_cached.shape, expected_shape)
        self.assertTrue(torch.equal(k_cached, k_new))

    def test_sliding_window(self):
        """Tests that the cache behaves like a ring buffer (sliding window)."""
        layer_idx, seq_len_first, seq_len_second = 0, 7, 5
        k1 = torch.randn(
            self.config.batch_size,
            self.config.num_kv_heads,
            seq_len_first,
            self.config.d_k,
        )
        v1 = torch.randn(
            self.config.batch_size,
            self.config.num_kv_heads,
            seq_len_first,
            self.config.d_k,
        )
        self.kv_cache.update(k1, v1, layer_idx)
        self.kv_cache.increment_pos(seq_len_first)
        self.assertEqual(self.kv_cache.current_pos, seq_len_first)
        k2 = torch.randn(
            self.config.batch_size,
            self.config.num_kv_heads,
            seq_len_second,
            self.config.d_k,
        )
        v2 = torch.randn(
            self.config.batch_size,
            self.config.num_kv_heads,
            seq_len_second,
            self.config.d_k,
        )
        self.kv_cache.update(k2, v2, layer_idx)
        self.kv_cache.increment_pos(seq_len_second)

        # The position should now be 12
        self.assertEqual(self.kv_cache.current_pos, seq_len_first + seq_len_second)

        k_cached, _ = self.kv_cache.get(layer_idx)

        # The total length of the cached sequence should not exceed max_seq_len
        self.assertEqual(k_cached.shape[2], self.config.max_seq_len)

        # Check that the cached data is correct
        k_expected = torch.cat([k1[:, :, -5:], k2[:, :, :5]], dim=2)
        self.assertTrue(torch.equal(k_cached, k_expected))

    def test_get_with_seq_len(self):
        """Tests that get(seq_len=...) includes tokens from a recent update."""
        layer_idx, seq_len = 0, 3
        k_new = torch.randn(
            self.config.batch_size, self.config.num_kv_heads, seq_len, self.config.d_k
        )
        v_new = torch.randn(
            self.config.batch_size, self.config.num_kv_heads, seq_len, self.config.d_k
        )

        # Update but DON'T increment_pos yet (simulating layer forward pass)
        self.kv_cache.update(k_new, v_new, layer_idx)

        # Regular get should be empty
        k_empty, _ = self.kv_cache.get(layer_idx)
        self.assertEqual(k_empty.shape[2], 0)

        # get with seq_len should have the new tokens
        k_full, _ = self.kv_cache.get(layer_idx, seq_len=seq_len)
        self.assertEqual(k_full.shape[2], seq_len)
        self.assertTrue(torch.equal(k_full, k_new))


if __name__ == "__main__":
    unittest.main()
