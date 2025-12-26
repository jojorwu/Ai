"""
Unit tests for the LongTermMemory module.
"""
import logging
import unittest

import numpy as np
from gradient_check import check_gradient, numerical_gradient

from nn_components.long_term_memory import LongTermMemory


class TestLongTermMemory(unittest.TestCase):
    """Tests for the LongTermMemory module."""

    def setUp(self):
        """Set up a simple LongTermMemory module for testing."""
        self.d_model = 16
        self.d_hidden = 32
        self.num_layers = 2
        self.ltm = LongTermMemory(self.d_model, self.d_hidden, self.num_layers)
        self.batch_size = 4
        self.input_data = np.random.randn(self.batch_size, 1, self.d_model)

    def test_forward_pass_shape(self):
        """Test the output shape of the forward pass."""
        logging.info("\nRunning Test: LTM Forward Pass Shape...")
        output = self.ltm.forward(self.input_data)
        self.assertEqual(output.shape, (self.batch_size, 1, self.d_model))
        logging.info("LTM Forward Pass Shape test PASSED.")

    def test_backward_pass_gradient(self):
        """Perform a numerical gradient check for the backward pass."""
        logging.info("\nRunning Test: LTM Backward Pass Gradient Check...")

        def forward_pass_for_grad_check(_):
            return self.ltm.forward(self.input_data)

        output = self.ltm.forward(self.input_data)
        dout = np.random.randn(*output.shape)
        self.ltm.backward(dout)

        trainable_params = self.ltm.get_trainable_params()
        for param_name, (param, analytical_grad) in trainable_params.items():
            if analytical_grad is None:
                continue

            numerical_grad_val = numerical_gradient(
                forward_pass_for_grad_check,
                param,
                dout
            )

            check_gradient(self, analytical_grad, numerical_grad_val, param_name)
        logging.info("LTM Backward Pass Gradient Check PASSED.")


if __name__ == '__main__':
    unittest.main()
