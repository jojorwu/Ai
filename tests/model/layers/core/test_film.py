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
