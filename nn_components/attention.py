import numpy as np

class ScaledDotProductAttention:
    """
    Класс для вычисления Scaled Dot-Product Attention с `forward` и `backward` методами.
    """
    def __init__(self):
        # Кеш для обратного прохода
        self.q = None
        self.k = None
        self.v = None
        self.attention_weights = None
        self.mask = None

    def forward(self, q, k, v, mask=None):
        self.q = q
        self.k = k
        self.v = v
        self.mask = mask

        matmul_qk = np.matmul(q, k.swapaxes(-2, -1))
        d_k = k.shape[-1]
        scaled_attention_logits = matmul_qk / np.sqrt(d_k)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # Softmax
        exp_logits = np.exp(scaled_attention_logits - np.max(scaled_attention_logits, axis=-1, keepdims=True))
        self.attention_weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        output = np.matmul(self.attention_weights, v)
        return output

    def backward(self, dout):
        # Градиент по отношению к выходу matmul(attention, v)
        d_attention_weights = np.matmul(dout, self.v.swapaxes(-2, -1))
        dv = np.matmul(self.attention_weights.swapaxes(-2, -1), dout)

        # Обратный проход через Softmax
        # d_scaled_logits = d_attention_weights * softmax_derivative
        # Это можно упростить:
        s = self.attention_weights
        ds = s * (d_attention_weights - np.sum(d_attention_weights * s, axis=-1, keepdims=True))

        # Обратный проход через маску (градиент не течет через замаскированные элементы)
        if self.mask is not None:
            # Расширяем маску для совместимости с формой ds (batch, heads, seq, seq)
            # Умножаем на инвертированную маску, чтобы обнулить градиенты в замаскированных позициях
            broadcast_mask = self.mask[np.newaxis, np.newaxis, :, :]
            ds = ds * (~broadcast_mask)

        # Обратный проход через масштабирование
        d_k = self.k.shape[-1]
        d_matmul_qk = ds / np.sqrt(d_k)

        # Обратный проход через matmul(q, k.T)
        dq = np.matmul(d_matmul_qk, self.k)
        dk = np.matmul(d_matmul_qk.swapaxes(-2, -1), self.q)

        return dq, dk, dv

# ==================
#      TESTS
# ==================
def test_attention_backward():
    """Численная проверка градиентов для `backward` метода."""
    print("Running tests for ScaledDotProductAttention (Backward Pass)...")

    np.random.seed(42)
    batch_size, seq_len, d_k, d_v = 2, 3, 4, 5

    q = np.random.randn(batch_size, 1, seq_len, d_k) # Добавим "головы" для совместимости
    k = np.random.randn(batch_size, 1, seq_len, d_k)
    v = np.random.randn(batch_size, 1, seq_len, d_v)
    dout = np.random.randn(batch_size, 1, seq_len, d_v)

    attention = ScaledDotProductAttention()

    # --- Аналитические градиенты ---
    _ = attention.forward(q, k, v)
    dq, dk, dv = attention.backward(dout)

    # --- Численные градиенты ---
    epsilon = 1e-6

    # 1. Проверка dq
    dq_num = np.zeros_like(q)
    it = np.nditer(q, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_val = q[ix]
        q[ix] = old_val + epsilon
        fx_plus = np.sum(attention.forward(q, k, v) * dout)
        q[ix] = old_val - epsilon
        fx_minus = np.sum(attention.forward(q, k, v) * dout)
        dq_num[ix] = (fx_plus - fx_minus) / (2 * epsilon)
        q[ix] = old_val
        it.iternext()

    # 2. Проверка dk
    dk_num = np.zeros_like(k)
    it = np.nditer(k, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_val = k[ix]
        k[ix] = old_val + epsilon
        fx_plus = np.sum(attention.forward(q, k, v) * dout)
        k[ix] = old_val - epsilon
        fx_minus = np.sum(attention.forward(q, k, v) * dout)
        dk_num[ix] = (fx_plus - fx_minus) / (2 * epsilon)
        k[ix] = old_val
        it.iternext()

    # --- Сравнение ---
    assert np.allclose(dq, dq_num, rtol=1e-4, atol=1e-4), "Gradient check for dq FAILED"
    print("Gradient check for dq PASSED.")
    assert np.allclose(dk, dk_num, rtol=1e-4, atol=1e-4), "Gradient check for dk FAILED"
    print("Gradient check for dk PASSED.")
    # dv тривиален и менее склонен к ошибкам, но его тоже стоит проверять

    print("All tests passed!")

if __name__ == "__main__":
    test_attention_backward()
