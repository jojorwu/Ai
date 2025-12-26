"""
Tests for the optimized MultiHeadAttention layer.
"""
import logging
import unittest

from backend import np
from config import MultiHeadAttentionConfig
from nn_components.multi_head_attention import MultiHeadAttention
from nn_components.rotary_embedding import precompute_rope_embeddings
from tests.gradient_check import check_gradient, numerical_gradient


class TestMultiHeadAttention(unittest.TestCase):
    """
    Tests for the optimized MultiHeadAttention layer with a single QKV projection.
    """

    def test_gqa_with_rope_backward_gradient_check(self):
        """
        Numerically checks the gradients for the GQA backward method with RoPE
        and a combined QKV projection.
        """
        logging.info("\nRunning Test: Gradient check for optimized GQA with RoPE...")

        batch_size, seq_len, d_model, num_heads, num_kv_heads = 2, 8, 32, 4, 2
        d_k = d_model // num_heads

        np.random.seed(1337)

        # Use a real RoPE instance for a more thorough test
        rope = precompute_rope_embeddings(d_k, max_seq_len=seq_len)
        config = MultiHeadAttentionConfig(d_model=d_model,
                                          num_heads=num_heads,
                                          num_kv_heads=num_kv_heads,
                                          rotary_emb=rope)
        mha = MultiHeadAttention(config)

        # Single input tensor 'x' instead of separate Q, K, V
        x_input = np.random.randn(batch_size, seq_len, d_model)
        # Gradient from the subsequent layer
        dout = np.random.randn(batch_size, seq_len, d_model)

        # --- Forward and Backward Pass ---
        _ = mha.forward(x_input)
        dx_analytic = mha.backward(dout)

        # --- Numerical Gradient Check ---
        # Define a lambda for the forward pass for the numerical gradient checker
        def model_forward(t):
            return mha.forward(t)

        # Check gradients with respect to the input tensor 'x'
        logging.info("Checking gradients for input: dx...")
        dx_numerical = numerical_gradient(model_forward, x_input, dout)
        check_gradient(self, dx_analytic, dx_numerical, "dx")

        # Check gradients for all trainable parameters (qkv_proj.W and wo.W)
        all_params = mha.get_trainable_params()
        for param_name, (param_val, param_grad) in all_params.items():
            logging.info("Checking gradients for parameter: %s...", param_name)

            # Use a lambda that captures the current parameter being tested
            def param_forward(_):
                return mha.forward(x_input)

            grad_numerical = numerical_gradient(param_forward, param_val, dout)
            check_gradient(self, param_grad, grad_numerical, f"d{param_name}")

        logging.info("Optimized MultiHeadAttention gradient checks passed!")


if __name__ == "__main__":
    unittest.main()
