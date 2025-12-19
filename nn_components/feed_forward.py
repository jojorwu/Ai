import numpy as np
from nn_components.linear import Linear
from nn_components.activations import SiLU

class FeedForward:
    """
    Реализация Feed-Forward Network с SwiGLU активацией.
    SwiGLU(x, W, V, W2) = (SiLU(x @ W) * (x @ V)) @ W2
    """
    def __init__(self, d_model, d_ff):
        # В SwiGLU d_ff часто имеет другое значение (e.g. 2/3 * 4 * d_model),
        # но для простоты мы оставим его настраиваемым, как и раньше.
        self.w1 = Linear(d_model, d_ff) # Corresponds to W
        self.w2 = Linear(d_ff, d_model) # Corresponds to W2
        self.w3 = Linear(d_model, d_ff) # Corresponds to V
        self.silu = SiLU()

        # Кеш для backward pass
        self.x_w1_activated = None
        self.x_w3 = None

    def get_params(self):
        """Возвращает словарь слоев."""
        return {'w1': self.w1, 'w2': self.w2, 'w3': self.w3}

    def forward(self, x):
        # Прогоняем вход через первые два линейных слоя
        x_w1 = self.w1.forward(x)
        self.x_w3 = self.w3.forward(x)

        # Применяем SiLU и гейт (поэлементное умножение)
        self.x_w1_activated = self.silu.forward(x_w1)
        gated = self.x_w1_activated * self.x_w3

        # Прогоняем через финальный линейный слой
        output = self.w2.forward(gated)

        return output

    def backward(self, dout):
        # Обратный проход через w2
        d_gated = self.w2.backward(dout)

        # Обратный проход через поэлементное умножение
        # dL/d(silu(x_w1)) = d_gated * x_w3
        # dL/d(x_w3) = d_gated * silu(x_w1)
        d_silu_x_w1 = d_gated * self.x_w3
        d_x_w3 = d_gated * self.x_w1_activated

        # Обратный проход через SiLU
        d_x_w1 = self.silu.backward(d_silu_x_w1)

        # Обратный проход через w1 и w3
        dx1 = self.w1.backward(d_x_w1)
        dx3 = self.w3.backward(d_x_w3)

        # Суммируем градиенты, так как x был входом для обоих путей
        return dx1 + dx3
