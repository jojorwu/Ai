import numpy as np
from nn_components.linear import Linear

class FeedForward:
    """
    Реализация Position-wise Feed-Forward Network.
    """
    def __init__(self, d_model, d_ff):
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.relu_cache = None

    def get_params(self):
        return self.linear1.get_params() + self.linear2.get_params()

    def relu(self, x):
        return np.maximum(0, x)

    def relu_backward(self, dout):
        dout[self.relu_cache <= 0] = 0
        return dout

    def forward(self, x):
        linear1_output = self.linear1.forward(x)
        relu_output = self.relu(linear1_output)
        self.relu_cache = linear1_output
        output = self.linear2.forward(relu_output)
        return output

    def backward(self, dout):
        d_relu_output = self.linear2.backward(dout)
        d_linear1_output = self.relu_backward(d_relu_output)
        dx = self.linear1.backward(d_linear1_output)
        return dx

# ==================
#      TESTS
# ==================
def test_feed_forward_backward():
    """Численная проверка градиентов для `backward` метода."""
    print("Running tests for FeedForward (Backward Pass)...")

    batch_size, seq_len, d_model, d_ff = 2, 3, 4, 8

    np.random.seed(42)
    ffn = FeedForward(d_model, d_ff)
    x = np.random.randn(batch_size, seq_len, d_model)
    dout = np.random.randn(batch_size, seq_len, d_model)

    _ = ffn.forward(x)
    dx = ffn.backward(dout)

    epsilon = 1e-6
    dx_num = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_val = x[ix]
        x[ix] = old_val + epsilon
        fx_plus = np.sum(ffn.forward(x) * dout)
        x[ix] = old_val - epsilon
        fx_minus = np.sum(ffn.forward(x) * dout)
        dx_num[ix] = (fx_plus - fx_minus) / (2 * epsilon)
        x[ix] = old_val
        it.iternext()

    assert np.allclose(dx, dx_num, rtol=1e-4, atol=1e-4), "Gradient check for dx FAILED"
    print("Gradient check for dx PASSED.")
    print("All tests passed!")

if __name__ == "__main__":
    test_feed_forward_backward()
