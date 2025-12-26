"""
Unit tests for the MixtureOfExperts module.
"""
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import unittest

import numpy as np

from gradient_check import check_gradient, numerical_gradient
from config import MoEConfig
from nn_components.moe import MixtureOfExperts


class TestMoE(unittest.TestCase):
    """Tests for the MixtureOfExperts module."""

    def setUp(self):
        """Set up a simple MoE layer for testing."""
        self.config = MoEConfig(d_model=16, d_ff=32, num_experts=4, top_k=2)
        self.moe = MixtureOfExperts(self.config)
        self.test_data = {
            "batch_size": 4,
            "seq_len": 8,
            "input":
            np.random.randn(4, 8, 16)
        }

    def test_forward_pass_shape(self):
        """Test the output shape of the forward pass."""
        logging.info("\nRunning Test: MoE Forward Pass Shape...")
        output, aux_loss = self.moe.forward(self.test_data["input"])
        self.assertEqual(output.shape, self.test_data["input"].shape)
        self.assertIsInstance(aux_loss, float)
        logging.info("MoE Forward Pass Shape test PASSED.")

    def test_backward_pass_gradient(self):
        """Perform a numerical gradient check for the backward pass."""
        logging.info("\nRunning Test: MoE Backward Pass Gradient Check...")

        output, _ = self.moe.forward(self.test_data["input"])
        dout = np.ones_like(output)
        self.moe.backward(dout)

        gate_params = self.moe.gate.get_trainable_params()
        for param_name, (param, analytical_grad) in gate_params.items():
            if analytical_grad is None:
                continue
            numerical_grad_val = numerical_gradient(
                lambda p_arg: self.moe.forward(self.test_data["input"])[0],
                param, dout)
            check_gradient(self, analytical_grad, numerical_grad_val,
                           f"gate.{param_name}")

        for i in range(self.config.num_experts):
            expert = self.moe.experts[i]
            for layer_name, layer_obj in expert.get_children().items():
                expert_params = layer_obj.get_trainable_params()
                for param_name, (param,
                                analytical_grad) in expert_params.items():
                    if analytical_grad is None:
                        continue
                    numerical_grad_val = numerical_gradient(
                        lambda p_arg: self.moe.forward(
                            self.test_data["input"])[0], param, dout)
                    check_gradient(
                        self, analytical_grad, numerical_grad_val,
                        f"expert_{i}.{layer_name}.{param_name}")

        logging.info("MoE Backward Pass Gradient Check PASSED.")


if __name__ == '__main__':
    unittest.main()
