"""
Tests for the PyTorch-based Rotary Positional Embedding (RoPE).
"""
import unittest

import torch

from src.model.layers.attention.rotary_embedding import (apply_rope_embeddings,
                                                precompute_rope_embeddings)


class TestRotaryEmbedding(unittest.TestCase):
    """
    Tests for the PyTorch RoPE implementation.
    """

    def test_precompute_embeddings_shape(self):
        """Tests the shape of the precomputed cosine and sine tensors."""
        d_k = 64
        max_seq_len = 128
        cos, sin = precompute_rope_embeddings(d_k, max_seq_len)

        # RoPE works on pairs, so the last dimension is d_k // 2 for complex numbers
        self.assertEqual(cos.shape, (max_seq_len, 1, d_k // 2))
        self.assertEqual(sin.shape, (max_seq_len, 1, d_k // 2))

    def test_apply_rope_embeddings_shape(self):
        """Tests that RoPE application preserves tensor shape."""
        batch, heads, seq_len, d_k = 4, 8, 16, 64
        x = torch.randn(batch, heads, seq_len, d_k)
        cos, sin = precompute_rope_embeddings(d_k, seq_len)

        x_rotated = apply_rope_embeddings(x, cos, sin)

        self.assertEqual(x_rotated.shape, x.shape)

    def test_backward_pass_computes_grads(self):
        """
        Tests that gradients flow correctly through the RoPE application.
        """
        batch, heads, seq_len, d_k = 4, 8, 16, 64
        x = torch.randn(batch, heads, seq_len, d_k, requires_grad=True)
        cos, sin = precompute_rope_embeddings(d_k, seq_len)

        # Forward pass
        x_rotated = apply_rope_embeddings(x, cos, sin)

        # Simulate a loss and backward pass
        fake_loss = x_rotated.sum()
        fake_loss.backward()

        # Check that gradients exist for the input tensor
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        # The gradient should not be all zeros
        self.assertFalse(torch.all(x.grad == 0))


if __name__ == "__main__":
    unittest.main()
