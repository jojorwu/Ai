import numpy as np
from nn_components.attention import ScaledDotProductAttention
from nn_components.linear import Linear

class MultiHeadAttention:
    """
    Реализация Multi-Head Attention слоя.
    """
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model должна делиться на num_heads без остатка."

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.wq = Linear(d_model, d_model)
        self.wk = Linear(d_model, d_model)
        self.wv = Linear(d_model, d_model)
        self.wo = Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def get_params(self):
        return self.wq.get_params() + self.wk.get_params() + self.wv.get_params() + self.wo.get_params()

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

    def forward(self, q, k, v, mask=None):
        q_proj = self.split_heads(self.wq.forward(q))
        k_proj = self.split_heads(self.wk.forward(k))
        v_proj = self.split_heads(self.wv.forward(v))

        scaled_attention = self.attention.forward(q_proj, k_proj, v_proj, mask)

        concat_attention = self.combine_heads(scaled_attention)

        output = self.wo.forward(concat_attention)

        return output

    def backward(self, dout):
        d_concat_attention = self.wo.backward(dout)

        d_scaled_attention = self.combine_heads_backward(d_concat_attention)

        dq_proj, dk_proj, dv_proj = self.attention.backward(d_scaled_attention)

        dq = self.wq.backward(self.split_heads_backward(dq_proj))
        dk = self.wk.backward(self.split_heads_backward(dk_proj))
        dv = self.wv.backward(self.split_heads_backward(dv_proj))

        return dq, dk, dv

    def split_heads_backward(self, x):
        batch_size, num_heads, seq_len, d_k = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

    def combine_heads_backward(self, x):
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

# ==================
#      TESTS
# ==================
def test_multi_head_attention_backward():
    """Численная проверка градиентов для `backward` метода."""
    print("Running tests for MultiHeadAttention (Backward Pass)...")

    batch_size, seq_len, d_model, num_heads = 2, 3, 4, 2

    np.random.seed(42)
    mha = MultiHeadAttention(d_model, num_heads)
    q = np.random.randn(batch_size, seq_len, d_model)
    k = np.random.randn(batch_size, seq_len, d_model)
    v = np.random.randn(batch_size, seq_len, d_model)
    dout = np.random.randn(batch_size, seq_len, d_model)

    _ = mha.forward(q, k, v)
    dq, dk, dv = mha.backward(dout)

    epsilon = 1e-6

    # Проверка dq
    dq_num = np.zeros_like(q)
    it = np.nditer(q, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_val = q[ix]
        q[ix] = old_val + epsilon
        fx_plus = np.sum(mha.forward(q, k, v) * dout)
        q[ix] = old_val - epsilon
        fx_minus = np.sum(mha.forward(q, k, v) * dout)
        dq_num[ix] = (fx_plus - fx_minus) / (2 * epsilon)
        q[ix] = old_val
        it.iternext()

    assert np.allclose(dq, dq_num, rtol=1e-4, atol=1e-4), "Gradient check for dq FAILED"
    print("Gradient check for dq PASSED.")

if __name__ == "__main__":
    test_multi_head_attention_backward()
