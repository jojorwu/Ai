import numpy as np
import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_components.kv_cache import KVCache

class TestKVCache(unittest.TestCase):
    def test_snapshot_and_restore(self):
        """Test the snapshot and restore functionality of the KVCache."""
        print("\nRunning Test: KVCache snapshot and restore...")
        num_layers = 2
        batch_size = 1
        num_kv_heads = 4
        d_k = 8
        max_seq_len = 16

        cache = KVCache(num_layers, batch_size, num_kv_heads, d_k, max_seq_len)

        # 1. Take an initial snapshot (should be all zeros)
        initial_snapshot = cache.snapshot()
        self.assertTrue(all(np.all(k == 0) for k, v in initial_snapshot))
        self.assertTrue(all(np.all(v == 0) for k, v in initial_snapshot))

        # 2. Update the cache
        k_update = np.random.randn(batch_size, num_kv_heads, 1, d_k)
        v_update = np.random.randn(batch_size, num_kv_heads, 1, d_k)
        cache.update(0, k_update, v_update, 0)

        # 3. Take a second snapshot
        updated_snapshot = cache.snapshot()
        self.assertFalse(np.all(updated_snapshot[0][0] == 0))
        self.assertFalse(np.all(updated_snapshot[0][1] == 0))

        # 4. Update the cache again
        k_update2 = np.random.randn(batch_size, num_kv_heads, 1, d_k)
        v_update2 = np.random.randn(batch_size, num_kv_heads, 1, d_k)
        cache.update(0, k_update2, v_update2, 1)

        # 5. Restore to the initial snapshot
        cache.restore(initial_snapshot)
        self.assertTrue(all(np.all(k == 0) for k, v in cache.cache))
        self.assertTrue(all(np.all(v == 0) for k, v in cache.cache))

        # 6. Restore to the updated snapshot
        cache.restore(updated_snapshot)
        self.assertTrue(np.allclose(cache.cache[0][0], updated_snapshot[0][0]))
        self.assertTrue(np.allclose(cache.cache[0][1], updated_snapshot[0][1]))

        print("KVCache snapshot and restore PASSED.")

if __name__ == "__main__":
    unittest.main()
