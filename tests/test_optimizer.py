"""
Tests for the `Adam` optimizer.
"""
import logging
import unittest

from backend import np
from config import OptimizerConfig
from nn_components.linear import Linear
from optimizer import Adam


class TestOptimizer(unittest.TestCase):
    """Tests for `Adam`."""

    def test_adam_optimizer(self):
        """Tests the Adam optimizer with a simple linear layer."""
        logging.info("\nRunning Test: Adam Optimizer...")

        input_size = 10
        output_size = 2

        config = OptimizerConfig(learning_rate=0.01, beta1=0.9, beta2=0.999,
                                 epsilon=1e-8, weight_decay=0.01, max_norm=1.0)
        linear_layer = Linear(input_size, output_size)
        optimizer = Adam(config)

        x = np.random.randn(1, input_size)
        d_out = np.random.randn(1, output_size)

        _ = linear_layer.forward(x)
        _ = linear_layer.backward(d_out)

        weights_before = np.copy(linear_layer.weights)
        optimizer.step(linear_layer.get_trainable_params())
        weights_after = linear_layer.weights

        self.assertFalse(np.array_equal(weights_before, weights_after),
                         "Optimizer step did not update weights.")
        logging.info("Adam Optimizer test passed.")


if __name__ == "__main__":
    unittest.main()
