"""
Implementation of the Scaled Dot-Product Attention mechanism.
"""
from backend import np

from nn_components.utils import softmax


class ScaledDotProductAttention:
    """
    Computes the Scaled Dot-Product Attention.
    """
    def __init__(self):
        self.d_k = None
        self.q = None
        self.k = None
        self.v = None
        self.attention_scores = None

    def forward(self, q, k, v, mask=None):
        """
        Performs the forward pass for Scaled Dot-Product Attention.
        q, k, v are expected to have shape (batch_size, num_heads, seq_len, d_k)
        """
        self.d_k = q.shape[-1]
        self.q, self.k, self.v = q, k, v

        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = np.where(mask, -np.inf, scores)

        self.attention_scores = softmax(scores)
        return self.attention_scores @ v

    def backward(self, dout):
        """
        Performs the backward pass for Scaled Dot-Product Attention.
        """
        # Gradients of the final output with respect to inputs
        dv = self.attention_scores.transpose(0, 1, 3, 2) @ dout
        d_attention_scores = dout @ self.v.transpose(0, 1, 3, 2)

        # Gradients of softmax
        # Note: In a real implementation, you'd use the Jacobian of softmax,
        # but for simplicity, we combine it with the score gradient calculation.
        d_scores = self.attention_scores * (
                d_attention_scores - np.sum(d_attention_scores * self.attention_scores,
                                            axis=-1, keepdims=True))

        # Gradients of scores with respect to Q and K
        d_scores /= np.sqrt(self.d_k)
        dq = d_scores @ self.k
        dk = d_scores.transpose(0, 1, 3, 2) @ self.q

        return dq, dk, dv
