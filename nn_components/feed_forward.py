import numpy as np
from nn_components.linear import Linear
from nn_components.activations import SiLU

class FeedForward:
    """
    Реализация Feed-Forward Network с SwiGLU активацией.
    """
    def __init__(self, d_model, d_ff):
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        self.silu = SiLU()
        self.x_w1_activated = None
        self.x_w3 = None

    def get_children(self):
        """Возвращает словарь дочерних слоев."""
        return {'w1': self.w1, 'w2': self.w2, 'w3': self.w3}

    def forward(self, x):
        x_w1 = self.w1.forward(x)
        self.x_w3 = self.w3.forward(x)
        self.x_w1_activated = self.silu.forward(x_w1)
        gated = self.x_w1_activated * self.x_w3
        output = self.w2.forward(gated)
        return output

    def backward(self, dout):
        d_gated = self.w2.backward(dout)
        d_silu_x_w1 = d_gated * self.x_w3
        d_x_w3 = d_gated * self.x_w1_activated
        d_x_w1 = self.silu.backward(d_silu_x_w1)
        dx1 = self.w1.backward(d_x_w1)
        dx3 = self.w3.backward(d_x_w3)
        return dx1 + dx3
