"""
Tests for activation functions.
"""
import unittest

import numpy as np

from nn_components.activations import Tanh


class TestActivations(unittest.TestCase):
    """
    Tests for activation functions.
    """

    def test_tanh_forward(self):
        """Test the forward pass of the Tanh activation function."""
        tanh = Tanh()
        x = np.array([-1, 0, 1])
        expected = np.tanh(x)
        self.assertTrue(np.allclose(tanh.forward(x), expected))

    def test_tanh_backward(self):
        """Test the backward pass of the Tanh activation function."""
        tanh = Tanh()
        x = np.array([-1, 0, 1])
        tanh.forward(x)
        dout = np.array([0.5, 1, 1.5])
        expected = dout * (1 - np.tanh(x) ** 2)
        self.assertTrue(np.allclose(tanh.backward(dout), expected))


if __name__ == "__main__":
    unittest.main()
