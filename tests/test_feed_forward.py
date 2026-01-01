"""
Tests for the PyTorch-based FeedForward (SwiGLU) layer.
"""
import unittest

import torch

from src.config import FeedForwardConfig
from src.model.layers.feed_forward import FeedForward


class TestFeedForward(unittest.TestCase):
    """
    Tests for the PyTorch FeedForward (SwiGLU) layer.
    """

    def test_forward_pass_shape(self):
        """Tests that the forward pass preserves the tensor shape."""
        config = FeedForwardConfig(d_model=64, d_ff=128)
        ffn = FeedForward(config)
        x = torch.randn(4, 10, config.d_model)  # Batch, SeqLen, Dim
        output = ffn(x)
        self.assertEqual(x.shape, output.shape)

    def test_backward_pass_computes_grads(self):
        """
        Tests the backward pass to ensure gradients are computed for all parameters.
        """
        config = FeedForwardConfig(d_model=64, d_ff=128, bias=True)
        ffn = FeedForward(config)
        x = torch.randn(4, 10, config.d_model, requires_grad=True)

        # Forward pass
        output = ffn(x)

        # Simulate a loss and backward pass
        fake_loss = output.sum()
        fake_loss.backward()

        # Check that gradients exist for all weights and biases
        self.assertIsNotNone(ffn.w1.weight.grad)
        self.assertIsNotNone(ffn.w1.bias.grad)
        self.assertIsNotNone(ffn.w2.weight.grad)
        self.assertIsNotNone(ffn.w2.bias.grad)
        self.assertIsNotNone(ffn.w3.weight.grad)
        self.assertIsNotNone(ffn.w3.bias.grad)

        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
