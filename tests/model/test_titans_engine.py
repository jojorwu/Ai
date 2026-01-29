"""
Unit tests for the TitansForwardEngine.
"""
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from src.model.titans_engine import TitansForwardEngine
from src.model.layers.decoder_block import ForwardPassInput


class TestTitansEngine(unittest.TestCase):
    """Tests for the TitansForwardEngine."""

    def setUp(self):
        self.mock_model = MagicMock(spec=nn.Module)
        # Setup nested config mock
        self.mock_model.config = MagicMock()
        self.mock_model.config.model = MagicMock()
        self.mock_model.config.model.gradient_checkpointing = False

        # Setup nested layers mock
        self.mock_model.layers = MagicMock()

        self.mock_model.training = False
        self.engine = TitansForwardEngine(self.mock_model)

    def test_process_sequence_basic(self):
        """Tests the basic sequence processing flow."""
        h = torch.randn(1, 10, 64)

        # Mock GatingNetwork
        mock_gating = MagicMock()
        mock_gating.return_value = (torch.tensor([2]), torch.tensor([1]))
        self.mock_model.layers.gating_network = mock_gating

        # Mock Decoder blocks
        mock_block1 = MagicMock()
        mock_block1.forward_direct.return_value = (torch.randn(1, 10, 64), torch.tensor(0.1))
        mock_block2 = MagicMock()
        mock_block2.forward_direct.return_value = (torch.randn(1, 10, 64), torch.tensor(0.2))
        self.mock_model.layers.decoder = [mock_block1, mock_block2]

        # Mock LTM
        self.mock_model.layers.long_term_memory = None

        out_h, total_aux = self.engine.process_sequence(h)

        self.assertEqual(out_h.shape, h.shape)
        self.assertAlmostEqual(total_aux.item(), 0.3)
        self.assertEqual(mock_gating.call_count, 1)
        self.assertEqual(mock_block1.forward_direct.call_count, 1)
        self.assertEqual(mock_block2.forward_direct.call_count, 1)

    def test_ltm_override(self):
        """Tests that LTM override is respected."""
        h = torch.randn(1, 1, 64)
        mock_ltm = MagicMock()
        mock_ltm.return_value = (torch.randn(1, 1, 64), None)

        # Mock GatingNetwork to skip decoder blocks for simplicity
        mock_gating = MagicMock()
        mock_gating.return_value = (torch.tensor([0]), torch.tensor([1]))
        self.mock_model.layers.gating_network = mock_gating

        self.engine.process_sequence(h, ltm_override=mock_ltm)

        mock_ltm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
