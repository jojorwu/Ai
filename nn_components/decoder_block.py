import numpy as np
from nn_components.multi_head_attention import MultiHeadAttention
from nn_components.feed_forward import FeedForward
from nn_components.layer_norm import LayerNormalization

class DecoderBlock:
    """
    Реализация одного блока декодера Трансформера.
    """
    def __init__(self, d_model, num_heads, d_ff):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

    def get_params(self):
        return self.mha.get_params() + self.ffn.get_params() + self.norm1.get_params() + self.norm2.get_params()

    def forward(self, x, mask=None):
        x_norm1 = self.norm1.forward(x)
        attn_output = self.mha.forward(q=x_norm1, k=x_norm1, v=x_norm1, mask=mask)
        x = x + attn_output

        x_norm2 = self.norm2.forward(x)
        ffn_output = self.ffn.forward(x_norm2)
        x = x + ffn_output

        return x

    def backward(self, dout):
        dffn_output = dout
        dx_residual2 = dout

        d_x_norm2 = self.ffn.backward(dffn_output)
        dx_from_norm2 = self.norm2.backward(d_x_norm2)

        dx_after_attn = dx_from_norm2 + dx_residual2

        d_attn_output = dx_after_attn
        dx_residual1 = dx_after_attn

        dq, dk, dv = self.mha.backward(d_attn_output)
        d_x_norm1 = dq + dk + dv
        dx_from_norm1 = self.norm1.backward(d_x_norm1)

        dx = dx_from_norm1 + dx_residual1

        return dx

# ==================
#      TESTS
# ==================
def test_decoder_block_backward():
    """Численная проверка градиентов для `backward` метода."""
    print("Running tests for DecoderBlock (Backward Pass)...")

    batch_size, seq_len, d_model, num_heads, d_ff = 2, 3, 4, 2, 8

    np.random.seed(42)
    block = DecoderBlock(d_model, num_heads, d_ff)
    x = np.random.randn(batch_size, seq_len, d_model)
    dout = np.random.randn(batch_size, seq_len, d_model)

    _ = block.forward(x)
    dx = block.backward(dout)

    epsilon = 1e-6
    dx_num = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_val = x[ix]
        x[ix] = old_val + epsilon
        fx_plus = np.sum(block.forward(x) * dout)
        x[ix] = old_val - epsilon
        fx_minus = np.sum(block.forward(x) * dout)
        dx_num[ix] = (fx_plus - fx_minus) / (2 * epsilon)
        x[ix] = old_val
        it.iternext()

    assert np.allclose(dx, dx_num, rtol=1e-4, atol=1e-4), "Gradient check for dx FAILED"
    print("Gradient check for dx PASSED.")
    print("All tests passed!")

if __name__ == "__main__":
    test_decoder_block_backward()
