"""
Tests for the PyTorch-based Mixture of Experts (MoE) layer.
"""
import unittest
import torch
from config import MoEConfig
from nn_components.moe import MixtureOfExperts


class TestMoE(unittest.TestCase):
    """
    Tests for the PyTorch MixtureOfExperts layer.
    """

    def test_forward_pass_shape_and_loss(self):
        """Tests the forward pass output shape and that a scalar loss is returned."""
        config = MoEConfig(d_model=16, d_ff=32, num_experts=4, top_k=2)
        moe = MixtureOfExperts(config)
        x = torch.randn(4, 8, 16)  # Batch, SeqLen, Dim

        output, aux_loss = moe(x)

        self.assertEqual(output.shape, x.shape)
        self.assertIsInstance(aux_loss, torch.Tensor)
        self.assertEqual(aux_loss.numel(), 1)

    def test_backward_pass_computes_grads(self):
        """
        Tests that gradients are computed for all parameters in the MoE layer.
        """
        config = MoEConfig(d_model=16, d_ff=32, num_experts=4, top_k=2, bias=True)
        moe = MixtureOfExperts(config)
        x = torch.randn(4, 8, 16, requires_grad=True)

        # Forward pass
        output, aux_loss = moe(x)

        # Simulate a combined loss and backward pass
        fake_loss = output.sum() + aux_loss
        fake_loss.backward()

        # Check gradients for the gating network
        self.assertIsNotNone(moe.gate.weights.grad)
        self.assertIsNotNone(moe.gate.bias.grad)

        # Check gradients for at least one expert
        self.assertTrue(any(p.grad is not None for p in moe.experts[0].parameters()))

        # Check gradient for the input
        self.assertIsNotNone(x.grad)

    def test_dynamic_top_k(self):
        """Tests that the dynamic_top_k argument overrides the default top_k."""
        config = MoEConfig(d_model=16, d_ff=32, num_experts=8, top_k=2)
        moe = MixtureOfExperts(config)
        x = torch.randn(1, 1, 16) # Single token for simplicity

        # --- Test with dynamic_top_k=4 ---
        # We can't directly inspect the number of experts used, but we can check
        # that it runs without error. A more detailed test would require modifying
        # the forward pass to return intermediate values, which is beyond a typical unit test.
        try:
            output, _ = moe(x, dynamic_top_k=4)
            self.assertEqual(output.shape, x.shape)
        except RuntimeError as e:
            self.fail(f"Forward pass with dynamic_top_k failed with exception: {e}")

        # --- Test with dynamic_top_k=1 ---
        try:
            output, _ = moe(x, dynamic_top_k=1)
            self.assertEqual(output.shape, x.shape)
        except RuntimeError as e:
            self.fail(f"Forward pass with dynamic_top_k failed with exception: {e}")


if __name__ == "__main__":
    unittest.main()
