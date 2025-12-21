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
