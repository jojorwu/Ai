import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimizer import Adam
from nn_components.linear import Linear

class TestOptimizer(unittest.TestCase):

    def test_adam_optimizer(self):
        """Tests the Adam optimizer with a simple linear layer."""
        print("\\nRunning Test: Adam Optimizer...")

        # Setup a simple linear layer
        input_size, output_size = 10, 5
        linear_layer = Linear(input_size, output_size)

        # Mock named parameters
        named_params = {'linear': linear_layer}

        # Initialize optimizer
        optimizer = Adam(named_params, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

        # Create some mock data and gradients
        mock_input = np.random.randn(1, input_size)
        mock_grad_output = np.random.randn(1, output_size)

        # Forward and backward pass to get gradients
        _ = linear_layer.forward(mock_input)
        _ = linear_layer.backward(mock_grad_output)

        # Store original weights
        original_W = np.copy(linear_layer.W)
        original_b = np.copy(linear_layer.b)

        # Perform one optimization step
        optimizer.step()

        # Check that weights have been updated
        self.assertFalse(np.array_equal(original_W, linear_layer.W), "Adam optimizer did not update weights W.")
        self.assertFalse(np.array_equal(original_b, linear_layer.b), "Adam optimizer did not update weights b.")

        print("Adam Optimizer test PASSED.")

if __name__ == '__main__':
    unittest.main()
