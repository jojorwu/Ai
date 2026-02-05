"""
Unit tests for the FiLMLayer.
"""
import unittest

import torch

from src.model.layers.core.film import FiLMLayer


class TestFiLMLayer(unittest.TestCase):
    """Tests for the FiLMLayer."""

    def test_forward_pass_shape(self):
        """Tests that the forward pass produces the correct output shape."""
        d_model = 64
        film = FiLMLayer(d_model)
        x = torch.randn(4, 10, d_model)
        ltm_state = torch.randn(4, 1, d_model)

        output = film(x, ltm_state)

        self.assertEqual(output.shape, x.shape)

    def test_backward_pass_computes_grads(self):
        """Tests that gradients are computed for all parameters."""
        d_model = 64
        film = FiLMLayer(d_model)
        x = torch.randn(4, 10, d_model, requires_grad=True)
        ltm_state = torch.randn(4, 1, d_model, requires_grad=True)

        output = film(x, ltm_state)
        fake_loss = output.sum()
        fake_loss.backward()

        self.assertIsNotNone(film.projection.weight.grad)
        self.assertIsNotNone(film.projection.bias.grad)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(ltm_state.grad)

    def test_gating_behavior(self):
        """Tests that gating is applied correctly."""
        d_model = 64
        film = FiLMLayer(d_model)
        x = torch.ones(1, 1, d_model)
        ltm_state = torch.randn(1, 1, d_model)

        # Manually set weights to produce specific gate value
        with torch.no_grad():
            # Project to [gamma, beta, gate]
            # Set gate_logits to a very large negative number -> gate close to 0
            film.projection.bias.fill_(0.0)
            # gate_logits are the last chunk of 3*d_model
            film.projection.bias[2*d_model:].fill_(-1000.0)

        output = film(x, ltm_state)
        # If gate is 0, output should be x
        self.assertTrue(torch.allclose(output, x, atol=1e-5))
