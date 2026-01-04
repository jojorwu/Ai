"""
Tests for the PyTorch-based Scaled Dot-Product Attention.
"""
import unittest
from unittest.mock import patch

import torch

from src.model.layers.attention import AttentionInput, ScaledDotProductAttention


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

    @unittest.skipIf(not torch.cuda.is_available(), "Flash Attention test requires CUDA")
    @patch('src.model.layers.attention.flash_attn_func')
    @patch('src.model.layers.attention.FLASH_ATTENTION_AVAILABLE', True)
    def test_uses_flash_attention_if_available(self, mock_flash_attn):
        """Tests that flash_attn_func is called when available and conditions are met."""
        attention = ScaledDotProductAttention()
        inputs = AttentionInput(
            q=torch.randn(1, 1, 4, 2, device='cuda', dtype=torch.float16),
            k=torch.randn(1, 1, 4, 2, device='cuda', dtype=torch.float16),
            v=torch.randn(1, 1, 4, 2, device='cuda', dtype=torch.float16),
            is_causal=True,
        )
        attention(inputs)
        mock_flash_attn.assert_called_once()

    @unittest.skipIf(not torch.cuda.is_available(), "Flash Attention test requires CUDA")
    @patch('torch.nn.functional.scaled_dot_product_attention')
    @patch('src.model.layers.attention.FLASH_ATTENTION_AVAILABLE', False)
    def test_uses_pytorch_fallback_if_flash_not_available(self, mock_pytorch_attn):
        """Tests that the PyTorch fallback is used when flash_attn is not available."""
        attention = ScaledDotProductAttention()
        inputs = AttentionInput(
            q=torch.randn(1, 1, 4, 2, device='cuda', dtype=torch.float16),
            k=torch.randn(1, 1, 4, 2, device='cuda', dtype=torch.float16),
            v=torch.randn(1, 1, 4, 2, device='cuda', dtype=torch.float16),
            is_causal=True,
        )
        attention(inputs)
        mock_pytorch_attn.assert_called_once()


if __name__ == "__main__":
    unittest.main()
