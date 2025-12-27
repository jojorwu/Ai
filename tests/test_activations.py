"""
Tests for PyTorch-based activation functions.
"""
import unittest
import torch

from nn_components.activations import Tanh


class TestActivations(unittest.TestCase):
    """
    Tests for PyTorch-based activation functions.
    """

    def test_tanh_forward(self):
        """Test the forward pass of the Tanh activation function."""
        tanh = Tanh()
        x = torch.tensor([-1.0, 0.0, 1.0])
        expected = torch.tanh(x)
        self.assertTrue(torch.allclose(tanh(x), expected))

    def test_tanh_backward(self):
        """Test the backward pass of the Tanh activation function using autograd."""
        tanh = Tanh()
        x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)

        # Forward pass
        output = tanh(x)

        # Simulate a loss and backward pass
        fake_loss = output.sum()
        fake_loss.backward()

        # Manually calculate expected gradients
        expected_grad = 1 - torch.tanh(x)**2

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.allclose(x.grad, expected_grad))


if __name__ == "__main__":
    unittest.main()
