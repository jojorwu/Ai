"""
Module containing the attention layer.
"""
from backend import np


class ScaledDotProductAttention:
    """
    Computes Scaled Dot-Product Attention with forward and backward passes.
    """

    def __init__(self):
        self.q = None
        self.k = None
        self.v = None
        self.attention_weights = None
        self.mask = None

    def forward(self, q, k, v, mask=None):
        """Performs the forward pass for Scaled Dot-Product Attention."""
        self.q, self.k, self.v, self.mask = q, k, v, mask

        matmul_qk = np.matmul(q, k.swapaxes(-2, -1))
        d_k = k.shape[-1]
        scaled_attention_logits = matmul_qk / np.sqrt(d_k)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        exp_logits = np.exp(scaled_attention_logits - np.max(scaled_attention_logits, axis=-1, keepdims=True))
        self.attention_weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return np.matmul(self.attention_weights, v)

    def backward(self, dout):
        """Performs the backward pass for Scaled Dot-Product Attention."""
        d_attention_weights = np.matmul(dout, self.v.swapaxes(-2, -1))
        dv = np.matmul(self.attention_weights.swapaxes(-2, -1), dout)

        s = self.attention_weights
        ds = s * (d_attention_weights - np.sum(d_attention_weights * s, axis=-1, keepdims=True))

        d_k = self.k.shape[-1]
        d_matmul_qk = ds / np.sqrt(d_k)

        dq = np.matmul(d_matmul_qk, self.k)
        dk = np.matmul(d_matmul_qk.swapaxes(-2, -1), self.q)

        return dq, dk, dv
