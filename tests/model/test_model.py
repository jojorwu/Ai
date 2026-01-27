"""
Unit tests for the Transformer model architecture.
"""
import unittest
from unittest.mock import MagicMock, patch

import torch

from src.config.core import (TrainConfig, TransformerConfig, ModelConfig,
                             VisionConfig, LTMArchitectureConfig)
from src.model.model import Transformer


class TestTransformer(unittest.TestCase):
    """Tests for the Transformer model."""

    def setUp(self):
        """Set up a mock model and config for testing."""
        config = TrainConfig.from_json('config/config_train.json')
        config.model.vocab_size = 50
        self.config = TransformerConfig(
            vocab_size=config.model.vocab_size,
            model=config.model,
            vision=config.vision,
            ltm=config.ltm,
            tokenizer=None,
        )
        self.model = Transformer(self.config)

    def test_layer_skipping(self):
        """
        Tests that the model correctly skips layers by mocking the GatingNetwork.
        """
        with patch.object(
            self.model.layers.gating_network,
            'forward',
            return_value=(torch.tensor([3]), torch.tensor([1])) # Mock: use 3 layers, 1 expert
        ) as mock_gate:
            from src.model.layers.decoder_block import DecoderBlock
            with patch.object(
                DecoderBlock, 'forward',
                return_value=(torch.randn(1, 10, 64), None)
            ) as mock_decoder:
                self.model(torch.randint(0, self.config.vocab_size, (1, 10)))
                mock_gate.assert_called_once()
                self.assertEqual(mock_decoder.call_count, 3)

    def test_dynamic_expert_allocation(self):
        """
        Tests that the model correctly allocates experts by mocking the GatingNetwork.
        """
        with patch.object(
            self.model.layers.gating_network,
            'forward',
            return_value=(torch.tensor([4]), torch.tensor([5])) # Mock: use 4 layers, 5 experts
        ) as mock_gate:
            from src.model.layers.decoder_block import DecoderBlock
            with patch.object(
                DecoderBlock, 'forward',
                return_value=(torch.randn(1, 10, 64), None)
            ) as mock_decoder:
                self.model(torch.randint(0, self.config.vocab_size, (1, 10)))
                mock_gate.assert_called_once()
                # The 'dynamic_top_k' is passed within the ForwardPassInput dataclass
                final_call_args = mock_decoder.call_args[0][0]
                self.assertEqual(final_call_args.dynamic_top_k, 5)

if __name__ == "__main__":
    unittest.main()
