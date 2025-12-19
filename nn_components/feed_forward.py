import numpy as np
from nn_components.linear import Linear
from nn_components.activations import GELU

class FeedForward:
    """
    Реализация Position-wise Feed-Forward Network с GELU активацией.
    """
    def __init__(self, d_model, d_ff):
        self.linear1 = Linear(d_model, d_ff)
        self.gelu = GELU()
        self.linear2 = Linear(d_ff, d_model)

    def get_params(self):
        """Возвращает словарь слоев."""
        # GELU не имеет обучаемых параметров
        return {'linear1': self.linear1, 'linear2': self.linear2}

    def forward(self, x):
        linear1_output = self.linear1.forward(x)
        gelu_output = self.gelu.forward(linear1_output)
        output = self.linear2.forward(gelu_output)
        return output

    def backward(self, dout):
        d_gelu_output = self.linear2.backward(dout)
        d_linear1_output = self.gelu.backward(d_gelu_output)
        dx = self.linear1.backward(d_linear1_output)
        return dx
