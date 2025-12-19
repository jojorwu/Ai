import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_components.multi_head_attention import MultiHeadAttention
from nn_components.rotary_embedding import RotaryPositionalEmbedding

def test_multi_head_attention_with_rope_backward_gradient_check():
    """Численная проверка градиентов для `backward` метода MultiHeadAttention с RoPE."""
    print("\nRunning Test: Gradient check for MultiHeadAttention with RoPE backward pass...")

    batch_size, seq_len, d_model, num_heads = 2, 6, 16, 4
    d_k = d_model // num_heads

    np.random.seed(1337)

    # Создаем и передаем RoPE
    rope = RotaryPositionalEmbedding(d_k, max_seq_len=seq_len)
    mha = MultiHeadAttention(d_model, num_heads, rotary_emb=rope)

    q = np.random.randn(batch_size, seq_len, d_model)
    k = np.random.randn(batch_size, seq_len, d_model)
    v = np.random.randn(batch_size, seq_len, d_model)
    dout = np.random.randn(batch_size, seq_len, d_model)

    # --- Аналитические градиенты ---
    _ = mha.forward(q, k, v)
    dq, dk, dv = mha.backward(dout)

    epsilon = 1e-5

    # --- Численная проверка градиентов для входов (dq, dk, dv) ---
    for name, x, dx_analytic in zip(['q', 'k', 'v'], [q, k, v], [dq, dk, dv]):
        print(f"Checking gradients for input: d{name}...")
        dx_numerical = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            original_value = x[ix]

            x[ix] = original_value + epsilon
            fx_plus_h = np.sum(mha.forward(q, k, v) * dout)

            x[ix] = original_value - epsilon
            fx_minus_h = np.sum(mha.forward(q, k, v) * dout)

            dx_numerical[ix] = (fx_plus_h - fx_minus_h) / (2 * epsilon)

            x[ix] = original_value
            it.iternext()

        assert np.allclose(dx_analytic, dx_numerical, rtol=1e-4, atol=1e-4), f"Gradient check for d{name} FAILED"
        print(f"Gradient check for d{name} PASSED.")

    # --- Численная проверка градиентов для весов (Wq, Wk, Wv, Wo) ---
    all_linear_layers = {'wq': mha.wq, 'wk': mha.wk, 'wv': mha.wv, 'wo': mha.wo}
    for layer_name, layer_obj in all_linear_layers.items():
        params = layer_obj.get_trainable_params()
        for p_name, (p_param, p_grad) in params.items():
            print(f"Checking gradients for parameter: {layer_name}.{p_name}...")

            grad_numerical = np.zeros_like(p_param)
            it = np.nditer(p_param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                original_value = p_param[ix]

                p_param[ix] = original_value + epsilon
                fx_plus_h = np.sum(mha.forward(q, k, v) * dout)

                p_param[ix] = original_value - epsilon
                fx_minus_h = np.sum(mha.forward(q, k, v) * dout)

                grad_numerical[ix] = (fx_plus_h - fx_minus_h) / (2 * epsilon)

                p_param[ix] = original_value
                it.iternext()

            assert np.allclose(p_grad, grad_numerical, rtol=1e-4, atol=1e-4), f"Gradient check for {layer_name}.{p_name} FAILED"
            print(f"Gradient check for parameter {layer_name}.{p_name} PASSED.")

    print("All MultiHeadAttention gradient checks passed!")

if __name__ == "__main__":
    test_multi_head_attention_with_rope_backward_gradient_check()
