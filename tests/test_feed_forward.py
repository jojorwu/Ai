"""
Tests for the PyTorch-based FeedForward (SwiGLU) layer.
"""
import sys
import os
import unittest
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn_components.feed_forward import FeedForward


class TestFeedForward(unittest.TestCase):
    """
    Tests for the PyTorch FeedForward (SwiGLU) layer.
    """

    def test_forward_pass_shape(self):
        """Tests that the forward pass preserves the tensor shape."""
        d_model, d_ff = 64, 128
        ffn = FeedForward(d_model, d_ff)
        x = torch.randn(4, 10, d_model)  # Batch, SeqLen, Dim
        output = ffn(x)
        self.assertEqual(x.shape, output.shape)

    def test_backward_pass_computes_grads(self):
        """
        Tests the backward pass to ensure gradients are computed for all parameters.
        """
        d_model, d_ff = 64, 128
        ffn = FeedForward(d_model, d_ff, bias=True)
        x = torch.randn(4, 10, d_model, requires_grad=True)

        # Forward pass
        output = ffn(x)

        # Simulate a loss and backward pass
        fake_loss = output.sum()
        fake_loss.backward()

        # Check that gradients exist for all weights and biases
        self.assertIsNotNone(ffn.w1.weights.grad)
        self.assertIsNotNone(ffn.w1.bias.grad)
        self.assertIsNotNone(ffn.w2.weights.grad)
        self.assertIsNotNone(ffn.w2.bias.grad)
        self.assertIsNotNone(ffn.w3.weights.grad)
        self.assertIsNotNone(ffn.w3.bias.grad)

        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
