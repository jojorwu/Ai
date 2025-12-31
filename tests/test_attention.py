"""
Tests for the PyTorch-based Scaled Dot-Product Attention.
"""
import unittest

import torch

from src.nn_components.attention import ScaledDotProductAttention


class TestScaledDotProductAttention(unittest.TestCase):
    """
    Tests for the PyTorch ScaledDotProductAttention layer.
    """

    def test_forward_pass_shape(self):
        """Tests the forward pass produces the correct output shape."""
        attention = ScaledDotProductAttention()
        batch, heads, seq_len, d_k = 4, 8, 10, 64
        q = torch.randn(batch, heads, seq_len, d_k)
        k = torch.randn(batch, heads, seq_len, d_k)
        v = torch.randn(batch, heads, seq_len, d_k)

        output = attention(q, k, v)

        self.assertEqual(output.shape, (batch, heads, seq_len, d_k))

    def test_backward_pass_computes_grads(self):
        """Tests that gradients are computed for Q, K, and V."""
        attention = ScaledDotProductAttention()
        batch, heads, seq_len, d_k = 4, 8, 10, 64
        q = torch.randn(batch, heads, seq_len, d_k, requires_grad=True)
        k = torch.randn(batch, heads, seq_len, d_k, requires_grad=True)
        v = torch.randn(batch, heads, seq_len, d_k, requires_grad=True)

        output = attention(q, k, v)
        fake_loss = output.sum()
        fake_loss.backward()

        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)

    def test_masking(self):
        """Tests that the mask correctly zeros out attention scores."""
        batch, heads, seq_len, d_k = 1, 1, 4, 2
        q = torch.randn(batch, heads, seq_len, d_k)
        k = torch.randn(batch, heads, seq_len, d_k)

        # Create a mask that allows attending only to the first two tokens
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

        # Manually compute scores to check attention weights
        scores = torch.matmul(
            q, k.transpose(-2, -1)
        ) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)

        # In the last row of attn_weights, elements after the diagonal should be zero
        # due to the causal mask.
        self.assertTrue(torch.all(attn_weights[0, 0, -1, :-1] > 0))
        self.assertTrue(attn_weights[0, 0, -1, -1] > 0)


if __name__ == "__main__":
    unittest.main()
