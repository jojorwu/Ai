"""
Tests for the PyTorch-based Linear layer.
"""
import unittest

import torch

from src.model.layers.linear import Linear


class TestLinear(unittest.TestCase):
    """
    Tests for the PyTorch Linear layer.
    """

    def test_forward_pass(self):
        """Tests the forward pass of the Linear layer."""
        input_dim, output_dim = 10, 20
        layer = Linear(input_dim, output_dim)

        # Test with a batch of vectors
        x_batch = torch.randn(32, input_dim)
        output = layer(x_batch)

        self.assertEqual(output.shape, (32, output_dim))

        # Test with a single vector
        x_single = torch.randn(input_dim)
        output_single = layer(x_single)
        self.assertEqual(output_single.shape, (output_dim,))


    def test_backward_pass_and_gradient_computation(self):
        """
        Tests the backward pass to ensure gradients are computed correctly by autograd.
        """
        input_dim, output_dim = 10, 20
        layer = Linear(input_dim, output_dim)

        x = torch.randn(32, input_dim, requires_grad=True)

        # Forward pass
        output = layer(x)

        # Simulate a loss and backward pass
        fake_loss = output.sum()
        fake_loss.backward()

        # Check that gradients exist for weights, bias, and input tensor
        self.assertIsNotNone(layer.weight.grad)
        self.assertEqual(layer.weight.grad.shape, layer.weight.shape)

        self.assertIsNotNone(layer.bias.grad)
        self.assertEqual(layer.bias.grad.shape, layer.bias.shape)

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


    def test_no_bias(self):
        """Tests that the layer works correctly when bias is disabled."""
        input_dim, output_dim = 10, 20
        layer = Linear(input_dim, output_dim, bias=False)

        self.assertIsNone(layer.bias)

        x = torch.randn(32, input_dim)
        output = layer(x)

        self.assertEqual(output.shape, (32, output_dim))

        # Ensure no gradient is computed for the bias
        fake_loss = output.sum()
        fake_loss.backward()
        self.assertIsNone(layer.bias)


if __name__ == "__main__":
    unittest.main()
