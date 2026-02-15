"""
Unit tests for the Transformer model architecture.
"""
import unittest
from unittest.mock import MagicMock, patch

import torch

from src.model.model import Transformer
from tests.test_utils import create_test_config


class TestTransformer(unittest.TestCase):
    """Tests for the Transformer model."""

    def setUp(self):
        """Set up a mock model and config for testing."""
        self.config = create_test_config(num_layers=4)
        self.model = Transformer(self.config.to_transformer_config())

    def test_layer_skipping(self):
        """
        Tests that the model correctly skips layers by mocking the GatingNetwork.
        """
        # Mock: use 3 layers, 1 expert
        with patch.object(
            self.model.layers.gating_network,
            'forward',
            return_value=(torch.tensor([3]), torch.tensor([1]))
        ) as mock_gate:
            from src.model.layers.blocks.decoder_block import DecoderBlock
            with patch.object(
                DecoderBlock, 'forward_direct',
                return_value=(torch.randn(1, 10, 16), None)
            ) as mock_decoder:
                self.model(torch.randint(0, self.config.model.vocab_size, (1, 10)))
                mock_gate.assert_called_once()
                self.assertEqual(mock_decoder.call_count, 3)

    def test_dynamic_expert_allocation(self):
        """
        Tests that the model correctly allocates experts by mocking the GatingNetwork.
        """
        # Mock: use 4 layers, 5 experts
        with patch.object(
            self.model.layers.gating_network,
            'forward',
            return_value=(torch.tensor([4]), torch.tensor([5]))
        ) as mock_gate:
            from src.model.layers.blocks.decoder_block import DecoderBlock
            with patch.object(
                DecoderBlock, 'forward_direct',
                return_value=(torch.randn(1, 10, 16), None)
            ) as mock_decoder:
                self.model(torch.randint(0, self.config.model.vocab_size, (1, 10)))
                mock_gate.assert_called_once()
                # The 'dynamic_top_k' is passed as the last argument to forward_direct
                # def forward_direct(self, x, ltm_state, kv_cache, layer_idx, dynamic_top_k)
                final_call_args = mock_decoder.call_args[0]
                self.assertEqual(final_call_args[4], 5)

    def test_logit_soft_clamping(self):
        """Tests that logits are soft-clamped when configured."""
        config = create_test_config()
        config.model.logit_soft_cap = 10.0
        model = Transformer(config.to_transformer_config())

        # Large inputs that would normally produce large logits
        x = torch.randint(0, config.model.vocab_size, (1, 5))

        # Force large weights to produce large logits
        with torch.no_grad():
            model.layers.embedding.embedding.weight.fill_(100.0)

        output = model(x)
        logits = output.logits

        # Logits should be bounded by approximately the soft_cap
        self.assertLessEqual(logits.abs().max().item(), 10.0 + 1e-4)

    def test_multimodal_kv_cache_no_redundancy(self):
        """Tests that vision tokens are not redundantly processed if already cached."""
        config = create_test_config()
        model = Transformer(config.to_transformer_config())

        # 1. First pass with images
        images = torch.randn(1, 3, 224, 224)
        x = torch.randint(0, config.model.vocab_size, (1, 5))

        from src.model.layers.attention.kv_cache import KVCache, KVCacheConfig
        d_k = config.model.d_model // config.model.num_heads
        kv_cache = KVCache(KVCacheConfig(
            num_layers=config.model.num_layers,
            batch_size=1,
            num_kv_heads=config.model.num_kv_heads,
            d_k=d_k,
            max_seq_len=1024
        ))

        # Initial forward
        model(x, images=images, kv_cache=kv_cache)
        pos_after_first = kv_cache.current_pos

        # 2. Second pass with same images and new token
        x2 = torch.randint(0, config.model.vocab_size, (1, 1))
        # Even if we pass images again, they should not be added to cache
        model(x2, images=images, kv_cache=kv_cache)

        # current_pos should only increase by 1 (the new text token)
        self.assertEqual(kv_cache.current_pos, pos_after_first + 1)

    def test_multimodal_error_raised(self):
        """Tests that MultimodalError is raised if images provided but no encoder."""
        config = create_test_config()
        model = Transformer(config.to_transformer_config())
        model.layers.vision_encoder = None

        images = torch.randn(1, 3, 224, 224)
        x = torch.randint(0, config.model.vocab_size, (1, 5))

        from src.utils.exceptions import MultimodalError
        with self.assertRaises(MultimodalError):
            model(x, images=images)

    def test_ltm_initialization_error(self):
        """Tests that InitializationError is raised on LTM failure."""
        config = create_test_config()
        # Directly pass incompatible values to the initializer bypass config validation
        # to test the error handling in ModelInitializer.
        t_config = config.to_transformer_config()
        t_config.model.ltm.d_hidden = 11
        t_config.model.ltm.num_heads = 4 # 11 is not divisible by 4
        t_config.model.ltm.num_layers = 1

        from src.utils.exceptions import InitializationError
        with self.assertRaises(InitializationError):
            Transformer(t_config)

if __name__ == "__main__":
    unittest.main()
