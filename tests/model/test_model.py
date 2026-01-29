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
            from src.model.layers.decoder_block import DecoderBlock
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
            from src.model.layers.decoder_block import DecoderBlock
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

if __name__ == "__main__":
    unittest.main()
