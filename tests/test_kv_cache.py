"""
Tests for the KVCache.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn_components.kv_cache import KVCache

class TestKVCache(unittest.TestCase):
    """
    Tests for the KVCache.
    """
    def setUp(self):
        self.num_layers = 2
        self.batch_size = 1
        self.num_kv_heads = 4
        self.d_k = 8
        self.max_seq_len = 16
        self.cache = KVCache(self.num_layers, self.batch_size, self.num_kv_heads, self.d_k, self.max_seq_len)
        print("\\nRunning Test: KVCache snapshot and restore...")

    def test_snapshot_and_restore(self):
        """Test the snapshot and restore functionality of the KVCache."""
        # 1. Initial state should be all zeros
        initial_snapshot = self.cache.snapshot()
        self.assertTrue(np.all(initial_snapshot['k_cache'] == 0))
        self.assertTrue(np.all(initial_snapshot['v_cache'] == 0))

        # 2. Update the cache with some data
        k_data = np.random.randn(self.batch_size, self.num_kv_heads, 5, self.d_k)
        v_data = np.random.randn(self.batch_size, self.num_kv_heads, 5, self.d_k)
        self.cache.update(k_data, v_data, layer_idx=0, seq_offset=0)

        # 3. Take a snapshot of the updated state
        updated_snapshot = self.cache.snapshot()

        # 4. Modify the cache again
        k_data2 = np.random.randn(self.batch_size, self.num_kv_heads, 3, self.d_k)
        v_data2 = np.random.randn(self.batch_size, self.num_kv_heads, 3, self.d_k)
        self.cache.update(k_data2, v_data2, layer_idx=1, seq_offset=2)

        # 5. Restore the cache to the updated_snapshot state
        self.cache.restore(updated_snapshot)

        # 6. Verify that the cache state matches the updated_snapshot
        np.testing.assert_array_equal(self.cache.k_cache, updated_snapshot['k_cache'])
        np.testing.assert_array_equal(self.cache.v_cache, updated_snapshot['v_cache'])

        # 7. Restore the initial state
        self.cache.restore(initial_snapshot)
        np.testing.assert_array_equal(self.cache.k_cache, initial_snapshot['k_cache'])
        np.testing.assert_array_equal(self.cache.v_cache, initial_snapshot['v_cache'])

        print("KVCache snapshot and restore PASSED.")

if __name__ == '__main__':
    unittest.main()
