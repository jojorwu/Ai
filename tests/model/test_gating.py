"""
Unit tests for the GatingNetwork.
"""
import unittest
import torch
from src.model.layers.gating import GatingNetwork


class TestGatingNetwork(unittest.TestCase):
    """Tests for the GatingNetwork class."""

    def test_forward_shape(self):
        """Tests the forward pass shapes."""
        d_model = 64
        num_layers = 12
        num_experts = 8
        gating = GatingNetwork(d_model, num_layers, num_experts)

        ltm_state = torch.randn(4, 1, d_model)
        active_layers, moe_top_k = gating(ltm_state)

        # active_layers shape: (batch, 1)
        self.assertEqual(active_layers.shape, (4, 1))
        # moe_top_k shape: (batch, 1)
        self.assertEqual(moe_top_k.shape, (4, 1))

    def test_output_values(self):
        """Tests that output values are within reasonable bounds."""
        d_model = 64
        num_layers = 12
        num_experts = 8
        gating = GatingNetwork(d_model, num_layers, num_experts)

        ltm_state = torch.randn(2, 1, d_model)
        active_layers, moe_top_k = gating(ltm_state)

        # active_layers should be between 1 and num_layers
        self.assertTrue(torch.all(active_layers >= 1))
        self.assertTrue(torch.all(active_layers <= num_layers))

        # moe_top_k should be between 1 and num_experts
        self.assertTrue(torch.all(moe_top_k >= 1))
        self.assertTrue(torch.all(moe_top_k <= num_experts))


if __name__ == "__main__":
    unittest.main()
