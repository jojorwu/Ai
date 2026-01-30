"""
Tests for the PyTorch-based RMSNorm layer.
"""
import math
import unittest

import torch

from src.model.layers.core.rms_norm import RMSNorm


class TestRMSNorm(unittest.TestCase):
    """
    Tests for the PyTorch RMSNorm layer.
    """

    def test_forward_pass_shape(self):
        """Tests that the forward pass preserves the tensor shape."""
        d_model = 64
        norm = RMSNorm(d_model)
        x = torch.randn(4, 10, d_model)  # Batch, SeqLen, Dim
        output = norm(x)
        self.assertEqual(x.shape, output.shape)

    def test_backward_pass_computes_grads(self):
        """
        Tests the backward pass to ensure gradients are computed for gamma and the input.
        """
        d_model = 64
        norm = RMSNorm(d_model)
        x = torch.randn(4, 10, d_model, requires_grad=True)

        # Forward pass
        output = norm(x)

        # Simulate a loss and backward pass
        fake_loss = output.sum()
        fake_loss.backward()

        # Check that gradients exist for gamma and the input tensor
        self.assertIsNotNone(norm.gamma.grad)
        self.assertEqual(norm.gamma.grad.shape, norm.gamma.shape)

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_normalization_effect(self):
        """
        Tests that the norm of the output (before scaling by gamma) is close to sqrt(d_model).
        """
        d_model = 64
        norm = RMSNorm(d_model)
        x = torch.randn(4, 10, d_model)

        # Manually perform the normalization part of the forward pass
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + norm.eps)
        normalized_x = x / rms

        # The L2 norm of each vector in the last dimension should be close to sqrt(d_model)
        # This is a property of RMSNorm.
        l2_norm = torch.linalg.norm(normalized_x, ord=2, dim=-1)  # pylint: disable=not-callable

        # We expect the norm to be close to sqrt(d_model)
        expected_norm = torch.full_like(l2_norm, fill_value=math.sqrt(d_model))

        self.assertTrue(torch.allclose(l2_norm, expected_norm, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
