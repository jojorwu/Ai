"""
Tests for the Adam optimizer.
"""

import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimizer import Adam
from nn_components.linear import Linear

class TestOptimizer(unittest.TestCase):
    """
    Tests for the Adam optimizer.
    """

    def test_adam_optimizer(self):
        """Tests the Adam optimizer with a simple linear layer."""
        print("\\nRunning Test: Adam Optimizer...")

        input_size, output_size = 10, 5
        linear_layer = Linear(input_size, output_size)

        named_params = {'linear': linear_layer}

        optimizer = Adam(named_params, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

        mock_input = np.random.randn(1, input_size)
        mock_grad_output = np.random.randn(1, output_size)

        _ = linear_layer.forward(mock_input)
        _ = linear_layer.backward(mock_grad_output)

        original_w = np.copy(linear_layer.W)
        original_b = np.copy(linear_layer.b)

        optimizer.step()

        self.assertFalse(np.array_equal(original_w, linear_layer.W),
                         "Adam optimizer did not update weights W.")
        self.assertFalse(np.array_equal(original_b, linear_layer.b),
                         "Adam optimizer did not update weights b.")

        print("Adam Optimizer test PASSED.")

if __name__ == '__main__':
    unittest.main()
