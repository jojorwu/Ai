"""
Tests for the KVCache.
"""
import logging
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
        logging.info("\nRunning Test: KVCache snapshot and restore...")

    def test_snapshot_and_restore(self):
        """Test the snapshot and restore functionality of the KVCache."""
        initial_snapshot = self.cache.snapshot()
        self.assertTrue(np.all(initial_snapshot['k_cache'] == 0))
        self.assertTrue(np.all(initial_snapshot['v_cache'] == 0))

        k_data = np.random.randn(self.batch_size, self.num_kv_heads, 5, self.d_k)
        v_data = np.random.randn(self.batch_size, self.num_kv_heads, 5, self.d_k)
        self.cache.update(k_data, v_data, 0, 0)

        updated_snapshot = self.cache.snapshot()

        k_data2 = np.random.randn(self.batch_size, self.num_kv_heads, 3, self.d_k)
        v_data2 = np.random.randn(self.batch_size, self.num_kv_heads, 3, self.d_k)
        self.cache.update(k_data2, v_data2, 1, 2)

        self.cache.restore(updated_snapshot)

        np.testing.assert_array_equal(self.cache.k_cache, updated_snapshot['k_cache'])
        np.testing.assert_array_equal(self.cache.v_cache, updated_snapshot['v_cache'])

        self.cache.restore(initial_snapshot)
        np.testing.assert_array_equal(self.cache.k_cache, initial_snapshot['k_cache'])
        np.testing.assert_array_equal(self.cache.v_cache, initial_snapshot['v_cache'])

        logging.info("KVCache snapshot and restore PASSED.")


if __name__ == '__main__':
    unittest.main()
