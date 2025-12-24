"""
Unit tests for the MixtureOfExperts module.
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn_components.moe import MixtureOfExperts
from tests.gradient_check import check_gradient, numerical_gradient

class TestMoE(unittest.TestCase):
    """Tests for the MixtureOfExperts module."""

    def setUp(self):
        """Set up a simple MoE layer for testing."""
        self.d_model = 16
        self.d_ff = 32
        self.num_experts = 4
        self.top_k = 2
        self.moe = MixtureOfExperts(self.d_model, self.d_ff, self.num_experts, self.top_k)
        self.batch_size = 4
        self.seq_len = 8
        self.input_data = np.random.randn(self.batch_size, self.seq_len, self.d_model)

    def test_forward_pass_shape(self):
        """Test the output shape of the forward pass."""
        print("\\nRunning Test: MoE Forward Pass Shape...")
        output, aux_loss = self.moe.forward(self.input_data)
        self.assertEqual(output.shape, self.input_data.shape)
        self.assertIsInstance(aux_loss, float)
        print("MoE Forward Pass Shape test PASSED.")

    def test_backward_pass_gradient(self):
        """Perform a numerical gradient check for the backward pass."""
        print("\\nRunning Test: MoE Backward Pass Gradient Check...")

        # The loss function for the gradient check will be the sum of the
        # main output plus the auxiliary loss.
        aux_loss_coeff = 0.01
        def forward_pass_for_grad_check():
            output, aux_loss = self.moe.forward(self.input_data)
            return np.sum(output) + aux_loss_coeff * aux_loss

        # Perform forward and backward pass to get analytical gradients.
        # The gradient of sum() is just 1s.
        output, aux_loss = self.moe.forward(self.input_data)
        dout = np.ones_like(output)
        self.moe.backward(dout)

        # --- Check Gate Gradients ---
        gate_params = self.moe.gate.get_trainable_params()
        for param_name, (param, analytical_grad) in gate_params.items():
            if analytical_grad is None: continue
            numerical_grad_val = numerical_gradient(
                lambda: self.moe.forward(self.input_data)[0], param, dout
            )
            check_gradient(self, analytical_grad, numerical_grad_val, f"gate.{param_name}")

        # --- Check Expert Gradients ---
        for i in range(self.num_experts):
            expert = self.moe.experts[i]
            for layer_name, layer_obj in expert.get_children().items():
                expert_params = layer_obj.get_trainable_params()
                for param_name, (param, analytical_grad) in expert_params.items():
                    if analytical_grad is None: continue
                    numerical_grad_val = numerical_gradient(
                        lambda: self.moe.forward(self.input_data)[0], param, dout
                    )
                    check_gradient(self, analytical_grad, numerical_grad_val, f"expert_{i}.{layer_name}.{param_name}")

        print("MoE Backward Pass Gradient Check PASSED.")

if __name__ == '__main__':
    unittest.main()
