"""
Targeted tests for the Gated Linear Associative Memory in LongTermMemory module.
"""
import unittest
import torch
from src.model.layers.titans.long_term_memory import LongTermMemory

class TestLTMGating(unittest.TestCase):
    """
    Verifies the input-dependent gating logic of the LongTermMemory module.
    """

    def setUp(self):
        self.d_model = 32
        self.d_hidden = 64
        self.num_heads = 4
        self.ltm = LongTermMemory(self.d_model, self.d_hidden, num_heads=self.num_heads)
        self.batch_size = 2

        self.head_dim = self.d_hidden // self.num_heads
        self.v_head_dim = self.d_model // self.num_heads

    def test_gating_mechanics(self):
        """Tests that different inputs produce different gate values and memory updates."""
        x1 = torch.randn(self.batch_size, 1, self.d_model)
        x2 = torch.randn(self.batch_size, 1, self.d_model)

        # Initial memory is zeros
        _, _, mem1 = self.ltm(x1)
        _, _, mem2 = self.ltm(x2)

        # Memory should be different for different inputs
        self.assertFalse(torch.allclose(mem1, mem2))

        # Verify shapes: [batch, heads, head_dim, v_head_dim]
        self.assertEqual(mem1.shape, (self.batch_size, self.num_heads, self.head_dim, self.v_head_dim))

    def test_persistence(self):
        """Tests that information persists across forward passes via the memory matrix."""
        x = torch.randn(self.batch_size, 1, self.d_model)

        # First pass
        context1, _, mem1 = self.ltm(x)

        # Second pass with same input but previous memory
        context2, _, mem2 = self.ltm(x, prev_mem=mem1)

        # Context and memory should change as they accumulate info
        self.assertFalse(torch.allclose(context1, context2))
        self.assertFalse(torch.allclose(mem1, mem2))

    def test_normalization_effect(self):
        """Tests that retrieval normalization keeps output magnitude bounded."""
        # Use very large input
        x = torch.randn(self.batch_size, 1, self.d_model) * 100.0

        context, _, _ = self.ltm(x)

        # Normalized output should have reasonable magnitude
        self.assertLess(context.abs().max(), 20.0)

    def test_learnability(self):
        """Tests that gates are learnable via backpropagation."""
        x = torch.randn(self.batch_size, 1, self.d_model, requires_grad=True)
        prev_mem = torch.randn(
            self.batch_size, self.num_heads, self.head_dim, self.v_head_dim, requires_grad=True
        )

        context, complexity, new_mem = self.ltm(x, prev_mem=prev_mem)

        loss = context.sum() + complexity.sum() + new_mem.sum()
        loss.backward()

        # Check that gradients flowed to gates and parameters
        self.assertIsNotNone(self.ltm.forget_gate_proj.weight.grad)
        self.assertIsNotNone(self.ltm.decay_bias.grad)
        self.assertIsNotNone(self.ltm.input_gate[0].weight.grad)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(prev_mem.grad)

if __name__ == "__main__":
    unittest.main()
