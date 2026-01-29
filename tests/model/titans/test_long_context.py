"""
Tests for long context stability (RoPE scaling, Sequential LTM updates).
"""
import unittest
import torch
from src.model.layers.attention.rotary_embedding import precompute_rope_embeddings, apply_rope_embeddings
from src.model.titans.titans_engine import TitansForwardEngine
from src.config.core import TrainConfig
from tests.test_utils import create_test_config
from unittest.mock import MagicMock

class TestLongContext(unittest.TestCase):
    """Verifies architectural stability for large token counts."""

    def test_ntk_rope_scaling_effect(self):
        """Tests that NTK scaling actually changes the embeddings."""
        d_k = 64
        max_seq_len = 1024

        # Standard RoPE
        cos1, sin1 = precompute_rope_embeddings(d_k, max_seq_len, ntk_factor=1.0)

        # NTK-scaled RoPE
        cos2, sin2 = precompute_rope_embeddings(d_k, max_seq_len, ntk_factor=2.0)

        # Embeddings should be different
        self.assertFalse(torch.allclose(cos1, cos2))
        self.assertFalse(torch.allclose(sin1, sin2))

    def test_apply_rope_oob_stability(self):
        """Tests that apply_rope_embeddings doesn't crash if offset is out of bounds."""
        d_k = 64
        max_seq_len = 10
        cos, sin = precompute_rope_embeddings(d_k, max_seq_len)

        x = torch.randn(1, 1, 5, d_k)
        # Offset + len = 15 > 10
        out = apply_rope_embeddings(x, cos, sin, seq_offset=10)

        # Should return something and not crash (identity fallback in my implementation)
        self.assertEqual(out.shape, x.shape)

    def test_sequential_ltm_update(self):
        """Tests that long sequences trigger chunked LTM updates in the engine."""
        mock_model = MagicMock()
        mock_model.config = create_test_config()
        mock_model.layers = MagicMock()

        # Setup LTM mock to track calls
        mock_ltm = MagicMock()
        mock_ltm.return_value = (torch.randn(1, 1, 64), torch.tensor([0.5]), torch.randn(1, 128, 64))
        mock_model.layers.long_term_memory = mock_ltm

        # Setup Gating mock
        mock_model.layers.gating_network.return_value = (torch.tensor([0]), torch.tensor([1]))

        engine = TitansForwardEngine(mock_model)

        # Sequence longer than chunk_size (512)
        h = torch.randn(1, 1500, 64)

        engine.process_sequence(h)

        # Should be called 3 times (ceil(1500 / 512))
        self.assertEqual(mock_ltm.call_count, 3)

if __name__ == "__main__":
    unittest.main()
