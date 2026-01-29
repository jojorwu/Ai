
import unittest
import torch
from src.model.layers.kv_cache import KVCache, KVCacheConfig
from src.agent.dataclasses import AgentState

class TestContextWindow(unittest.TestCase):
    def test_anchor_aware_kv_cache(self):
        config = KVCacheConfig(
            num_layers=1,
            batch_size=1,
            num_kv_heads=1,
            d_k=8,
            max_seq_len=10,
            anchor_size=2
        )
        cache = KVCache(config)

        # 1. Fill cache completely (10 tokens)
        for i in range(10):
            k = torch.full((1, 1, 1, 8), float(i))
            v = torch.full((1, 1, 1, 8), float(i))
            cache.update(k, v, layer_idx=0)
            cache.increment_pos(1)

        # Verify chronological order
        k_out, _ = cache.get(layer_idx=0)
        expected = torch.arange(10).view(1, 1, 10, 1).expand(-1, -1, -1, 8).float()
        self.assertTrue(torch.allclose(k_out, expected))

        # 2. Add 11th token. Should overwrite token at index 2 (first after anchors)
        k = torch.full((1, 1, 1, 8), 10.0)
        v = torch.full((1, 1, 1, 8), 10.0)
        cache.update(k, v, layer_idx=0)
        cache.increment_pos(1)

        k_out, _ = cache.get(layer_idx=0)
        # Expected: Anchors [0, 1] + Sliding [3, 4, 5, 6, 7, 8, 9, 10]
        expected_indices = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
        expected = torch.tensor(expected_indices).view(1, 1, 10, 1).expand(-1, -1, -1, 8).float()
        self.assertTrue(torch.allclose(k_out, expected))

    def test_agent_state_pruning(self):
        config = KVCacheConfig(
            num_layers=1,
            batch_size=1,
            num_kv_heads=1,
            d_k=8,
            max_seq_len=10,
            anchor_size=2
        )
        cache = KVCache(config)
        state = AgentState(
            conversation_history_tokens=list(range(10)),
            main_cache=cache,
            _processed_count=10
        )
        # Simulate that all 10 tokens are in cache
        cache.current_pos = 10

        # Prune to 6 tokens
        state.prune_history(context_window_size=6)

        # Expected: anchors [0, 1] + recent [6, 7, 8, 9]
        self.assertEqual(state.conversation_history_tokens, [0, 1, 6, 7, 8, 9])
        # Cache position should NOT be reset
        self.assertEqual(cache.current_pos, 10)

        # get_new_tokens should be empty
        self.assertEqual(state.get_new_tokens(), [])

        # Add new token
        state.append_tokens([10])
        self.assertEqual(state.get_new_tokens(), [10])

if __name__ == '__main__':
    unittest.main()
