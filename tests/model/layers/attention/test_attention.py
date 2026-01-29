"""
Tests for the PyTorch-based Scaled Dot-Product Attention.
"""
import unittest
from unittest.mock import patch

import torch

from src.model.layers.attention.attention import AttentionInput, ScaledDotProductAttention


class TestScaledDotProductAttention(unittest.TestCase):
    """
    Tests for the PyTorch ScaledDotProductAttention layer.
    """

    def test_forward_pass_shape(self):
        """Tests the forward pass produces the correct output shape."""
        attention = ScaledDotProductAttention()
        batch, heads, seq_len, d_k = 4, 8, 10, 64
        inputs = AttentionInput(
            q=torch.randn(batch, heads, seq_len, d_k),
            k=torch.randn(batch, heads, seq_len, d_k),
            v=torch.randn(batch, heads, seq_len, d_k),
        )
        output = attention(inputs)
        self.assertEqual(output.shape, (batch, heads, seq_len, d_k))

    def test_backward_pass_computes_grads(self):
        """Tests that gradients are computed for Q, K, and V."""
        attention = ScaledDotProductAttention()
        batch, heads, seq_len, d_k = 4, 8, 10, 64
        inputs = AttentionInput(
            q=torch.randn(batch, heads, seq_len, d_k, requires_grad=True),
            k=torch.randn(batch, heads, seq_len, d_k, requires_grad=True),
            v=torch.randn(batch, heads, seq_len, d_k, requires_grad=True),
        )
        output = attention(inputs)
        fake_loss = output.sum()
        fake_loss.backward()

        self.assertIsNotNone(inputs.q.grad)
        self.assertIsNotNone(inputs.k.grad)
        self.assertIsNotNone(inputs.v.grad)

    @patch('torch.nn.functional.scaled_dot_product_attention')
    def test_causal_masking(self, mock_attention):
        """Tests that is_causal is correctly passed to the backend."""
        attention = ScaledDotProductAttention()
        batch, heads, seq_len, d_k = 1, 1, 4, 2
        inputs = AttentionInput(
            q=torch.randn(batch, heads, seq_len, d_k),
            k=torch.randn(batch, heads, seq_len, d_k),
            v=torch.randn(batch, heads, seq_len, d_k),
            is_causal=True,
        )
        attention(inputs)
        # Check if the backend function was called with is_causal=True
        mock_attention.assert_called_with(
            inputs.q, inputs.k, inputs.v, attn_mask=None, is_causal=True
        )


if __name__ == "__main__":
    unittest.main()
