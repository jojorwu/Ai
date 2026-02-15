"""
Tests for the PyTorch-based LongTermMemory module.
"""
import unittest

import torch

from src.model.layers.titans.long_term_memory import LongTermMemory


class TestLongTermMemory(unittest.TestCase):
    """
    Tests for the PyTorch LongTermMemory module.
    """

    def test_forward_pass_shape(self):
        """Tests that the forward pass produces the correct output shape."""
        d_model, d_hidden, num_layers, num_heads = 64, 128, 2, 4
        ltm = LongTermMemory(d_model, d_hidden, num_layers, num_heads=num_heads)

        x = torch.randn(4, 1, d_model)  # Batch, SeqLen (1 for summary), Dim
        context, complexity_score, new_mem = ltm(x)

        self.assertEqual(context.shape, x.shape)
        self.assertEqual(complexity_score.shape, (4, 1, 1))

        # New shape: [batch, num_heads, head_dim, v_head_dim]
        # head_dim = d_hidden // num_heads = 128 // 4 = 32
        # v_head_dim = d_model // num_heads = 64 // 4 = 16
        self.assertEqual(new_mem.shape, (4, 4, 32, 16))

    def test_backward_pass_computes_grads(self):
        """
        Tests that gradients are computed for all parameters in the LTM.
        """
        d_model, d_hidden, num_layers, num_heads = 64, 128, 2, 4
        ltm = LongTermMemory(d_model, d_hidden, num_layers, num_heads=num_heads)

        x = torch.randn(4, 1, d_model, requires_grad=True)
        # Provide prev_mem to ensure memory_decay gradient is non-zero
        prev_mem = torch.randn(4, 4, 32, 16, requires_grad=True)

        # Forward pass
        context, complexity_score, _ = ltm(x, prev_mem=prev_mem)

        # Simulate a loss and backward pass
        fake_loss = context.sum() + complexity_score.sum()
        fake_loss.backward()

        # Check that gradients exist for all parameters in the network
        for param in ltm.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.all(param.grad == 0))

        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
