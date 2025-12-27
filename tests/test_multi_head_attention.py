"""
Tests for the optimized MultiHeadAttention layer.
"""
import logging
import unittest

from tests.gradient_check import check_gradient, numerical_gradient

from backend import np
from config import MultiHeadAttentionConfig
from nn_components.multi_head_attention import MultiHeadAttention
from nn_components.rotary_embedding import precompute_rope_embeddings


class TestMultiHeadAttention(unittest.TestCase):
    """
    Tests for the optimized MultiHeadAttention layer with a single QKV projection.
    """

    def _setup_test(self):
        """Sets up the test data and MultiHeadAttention layer."""
        batch_size, seq_len, d_model, num_heads, num_kv_heads = 2, 8, 32, 4, 2
        d_k = d_model // num_heads
        np.random.seed(1337)
        rope = precompute_rope_embeddings(d_k, max_seq_len=seq_len)
        config = MultiHeadAttentionConfig(d_model=d_model,
                                          num_heads=num_heads,
                                          num_kv_heads=num_kv_heads,
                                          rotary_emb=rope)
        mha = MultiHeadAttention(config)
        x_input = np.random.randn(batch_size, seq_len, d_model)
        dout = np.random.randn(batch_size, seq_len, d_model)
        return mha, x_input, dout

    def _check_gradients(self, mha, x_input, dout):
        """Checks the gradients for the MultiHeadAttention layer."""
        _ = mha.forward(x_input)
        dx_analytic = mha.backward(dout)

        def model_forward(t):
            return mha.forward(t)

        dx_numerical = numerical_gradient(model_forward, x_input, dout)
        check_gradient(self, dx_analytic, dx_numerical, "dx")

        all_params = mha.get_trainable_params()
        for param_name, (param_val, param_grad) in all_params.items():
            def param_forward(_):
                return mha.forward(x_input)

            grad_numerical = numerical_gradient(param_forward, param_val, dout)
            check_gradient(self, param_grad, grad_numerical, f"d{param_name}")

    def test_gqa_with_rope_backward_gradient_check(self):
        """
        Numerically checks the gradients for the GQA backward method with RoPE
        and a combined QKV projection.
        """
        logging.info("\nRunning Test: Gradient check for optimized GQA with RoPE...")
        mha, x_input, dout = self._setup_test()
        self._check_gradients(mha, x_input, dout)
        logging.info("Optimized MultiHeadAttention gradient checks passed!")


if __name__ == "__main__":
    unittest.main()
