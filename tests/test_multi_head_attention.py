"""
Tests for the MultiHeadAttention layer.
"""
import logging
import unittest

import numpy as np

from nn_components.multi_head_attention import MultiHeadAttention
from nn_components.rotary_embedding import RotaryPositionalEmbedding
from tests.gradient_check import check_gradient, numerical_gradient


class TestMultiHeadAttention(unittest.TestCase):
    """
    Tests for the MultiHeadAttention layer.
    """

    def test_gqa_with_rope_backward_gradient_check(self):
        """Numerically checks the gradients for the GQA backward method with RoPE."""
        logging.info("\nRunning Test: Gradient check for GQA with RoPE backward pass...")

        batch_size, seq_len, d_model, num_heads, num_kv_heads = 2, 6, 32, 8, 2
        d_k = d_model // num_heads

        np.random.seed(1337)

        rope = RotaryPositionalEmbedding(d_k, max_seq_len=seq_len)
        mha = MultiHeadAttention(d_model, num_heads, num_kv_heads, rotary_emb=rope)

        q = np.random.randn(batch_size, seq_len, d_model)
        k = np.random.randn(batch_size, seq_len, d_model)
        v = np.random.randn(batch_size, seq_len, d_model)
        dout = np.random.randn(batch_size, seq_len, d_model)

        _ = mha.forward(q, k, v)
        dq, dk, dv = mha.backward(dout)

        model_forward = lambda: mha.forward(q, k, v)

        for name, x, dx_analytic in zip(['q', 'k', 'v'], [q, k, v], [dq, dk, dv]):
            logging.info(f"Checking gradients for input: d{name}...")
            dx_numerical = numerical_gradient(model_forward, x, dout)
            check_gradient(self, dx_analytic, dx_numerical, f"d{name}")

        all_linear_layers = {'wq': mha.wq, 'wk': mha.wk, 'wv': mha.wv, 'wo': mha.wo}
        for layer_name, layer_obj in all_linear_layers.items():
            params = layer_obj.get_trainable_params()
            for p_name, (p_param, p_grad) in params.items():
                logging.info(f"Checking gradients for parameter: {layer_name}.{p_name}...")
                grad_numerical = numerical_gradient(model_forward, p_param, dout)
                check_gradient(self, p_grad, grad_numerical, f"parameter {layer_name}.{p_name}")

        logging.info("All MultiHeadAttention gradient checks passed!")


if __name__ == "__main__":
    unittest.main()
