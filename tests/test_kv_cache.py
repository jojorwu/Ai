"""
Tests for the KVCache.
"""
import unittest

import numpy as np

from nn_components.kv_cache import KVCache, KVCacheConfig


class TestKVCache(unittest.TestCase):
    """
    Tests for the KVCache.
    """

    def setUp(self):
        """Set up the test environment."""
        self.num_layers = 2
        self.batch_size = 1
        self.num_kv_heads = 4
        self.d_k = 8
        self.max_seq_len = 16
        config = KVCacheConfig(
            num_layers=self.num_layers,
            batch_size=self.batch_size,
            num_kv_heads=self.num_kv_heads,
            d_k=self.d_k,
            max_seq_len=self.max_seq_len
        )
        self.cache = KVCache(config)

    def test_update_and_get(self):
        """Test the basic update and get functionality."""
        k_data = np.random.randn(self.batch_size, self.num_kv_heads, 5, self.d_k)
        v_data = np.random.randn(self.batch_size, self.num_kv_heads, 5, self.d_k)
        self.cache.update(k_data, v_data, 0)

        k_cached, v_cached = self.cache.get(0)
        self.assertEqual(k_cached.shape[-2], 5)
        np.testing.assert_array_equal(k_cached, k_data)
        np.testing.assert_array_equal(v_cached, v_data)
        np.testing.assert_array_equal(v_cached, v_data)

    def test_ring_buffer_wrapping(self):
        """Test that the ring buffer wraps around correctly."""
        # Fill the cache completely
        k_full = np.random.randn(self.batch_size, self.num_kv_heads, self.max_seq_len, self.d_k)
        v_full = np.random.randn(self.batch_size, self.num_kv_heads, self.max_seq_len, self.d_k)
        self.cache.update(k_full, v_full, 0)
        self.assertTrue(self.cache.is_filled)
        self.assertEqual(self.cache.current_pos, 0)

        # Update with new data that should wrap
        k_new = np.random.randn(self.batch_size, self.num_kv_heads, 4, self.d_k)
        v_new = np.random.randn(self.batch_size, self.num_kv_heads, 4, self.d_k)
        self.cache.update(k_new, v_new, 0)
        self.assertEqual(self.cache.current_pos, 4)

        # Verify that the retrieved data is correctly ordered
        k_cached, v_cached = self.cache.get(0)
        self.assertEqual(k_cached.shape[-2], self.max_seq_len)

        # The last 12 elements should be from the original k_full
        np.testing.assert_array_equal(k_cached[..., :12, :], k_full[..., 4:, :])
        # The first 4 elements should be the new data
        np.testing.assert_array_equal(k_cached[..., 12:, :], k_new)


    def test_snapshot_and_restore(self):
        """Test the snapshot and restore functionality of the KVCache."""
        initial_snapshot = self.cache.snapshot()
        self.assertTrue(np.all(initial_snapshot['k_cache'] == 0))
        self.assertTrue(np.all(initial_snapshot['v_cache'] == 0))
        self.assertEqual(initial_snapshot['current_pos'], 0)

        k_data = np.random.randn(self.batch_size, self.num_kv_heads, 5, self.d_k)
        v_data = np.random.randn(self.batch_size, self.num_kv_heads, 5, self.d_k)
        self.cache.update(k_data, v_data, 0)

        updated_snapshot = self.cache.snapshot()
        self.assertEqual(updated_snapshot['current_pos'], 5)


        k_data2 = np.random.randn(self.batch_size, self.num_kv_heads, 3, self.d_k)
        v_data2 = np.random.randn(self.batch_size, self.num_kv_heads, 3, self.d_k)
        self.cache.update(k_data2, v_data2, 1)

        self.cache.restore(updated_snapshot)

        np.testing.assert_array_equal(self.cache.k_cache, updated_snapshot['k_cache'])
        np.testing.assert_array_equal(self.cache.v_cache, updated_snapshot['v_cache'])
        self.assertEqual(self.cache.current_pos, updated_snapshot['current_pos'])


        self.cache.restore(initial_snapshot)
        np.testing.assert_array_equal(self.cache.k_cache, initial_snapshot['k_cache'])
        np.testing.assert_array_equal(self.cache.v_cache, initial_snapshot['v_cache'])
        self.assertEqual(self.cache.current_pos, initial_snapshot['current_pos'])


if __name__ == '__main__':
    unittest.main()
