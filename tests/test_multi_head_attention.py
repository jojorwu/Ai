"""
Tests for the PyTorch-based MultiHeadAttention layer.
"""
import unittest

import torch

from config import MultiHeadAttentionConfig
from nn_components.multi_head_attention import MultiHeadAttention
from nn_components.rotary_embedding import precompute_rope_embeddings


class TestMultiHeadAttention(unittest.TestCase):
    """
    Tests for the PyTorch MultiHeadAttention layer.
    """

    def test_forward_pass_shape(self):
        """Tests that the forward pass preserves the tensor shape."""
        d_model, num_heads, num_kv_heads = 64, 4, 2
        config = MultiHeadAttentionConfig(
            d_model=d_model, num_heads=num_heads, num_kv_heads=num_kv_heads
        )
        mha = MultiHeadAttention(config)

        x = torch.randn(4, 10, d_model)  # Batch, SeqLen, Dim
        output = mha(x)

        self.assertEqual(x.shape, output.shape)

    def test_backward_pass_computes_grads(self):
        """
        Tests that gradients are computed for all parameters in the MHA layer.
        """
        d_model, num_heads, num_kv_heads = 64, 4, 2
        rope = precompute_rope_embeddings(d_model // num_heads, 10)
        config = MultiHeadAttentionConfig(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rotary_emb=rope,
            bias=True
        )
        mha = MultiHeadAttention(config)

        x = torch.randn(4, 10, d_model, requires_grad=True)

        # Forward pass
        output = mha(x)

        # Simulate a loss and backward pass
        fake_loss = output.sum()
        fake_loss.backward()

        # Check that gradients exist for the projection weights and biases
        self.assertIsNotNone(mha.qkv_proj.weights.grad)
        self.assertIsNotNone(mha.qkv_proj.bias.grad)
        self.assertIsNotNone(mha.wo.weights.grad)
        self.assertIsNotNone(mha.wo.bias.grad)

        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
