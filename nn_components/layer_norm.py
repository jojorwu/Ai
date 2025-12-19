import numpy as np

class LayerNormalization:
    """
    Реализация Layer Normalization.
    Нормализует активации по последней размерности (features).
    """
    def __init__(self, d_model, epsilon=1e-5):
        """
        Инициализация слоя.

        Args:
            d_model (int): Размерность модели (размер вектора признаков).
            epsilon (float): Небольшое значение для предотвращения деления на ноль.
        """
        self.d_model = d_model
        self.epsilon = epsilon
        # Обучаемые параметры: gamma (масштаб) и beta (сдвиг)
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

        # Кеш для обратного прохода
        self.x_normalized = None
        self.x_mean = None
        self.x_std = None
        self.x = None
        self.dgamma = None
        self.dbeta = None

    def get_params(self):
        return [self]

    def get_trainable_params(self):
        return {'gamma': (self.gamma, self.dgamma), 'beta': (self.beta, self.dbeta)}

    def forward(self, x):
        """
        Прямой проход для Layer Normalization.

        Args:
            x (np.ndarray): Входной тензор (размер: ..., d_model).

        Returns:
            np.ndarray: Нормализованный тензор того же размера.
        """
        self.x = x
        # 1. Вычислить среднее и стандартное отклонение по последней оси
        self.x_mean = np.mean(x, axis=-1, keepdims=True)
        self.x_std = np.std(x, axis=-1, keepdims=True)

        # 2. Нормализовать x
        self.x_normalized = (x - self.x_mean) / (self.x_std + self.epsilon)

        # 3. Применить масштабирование и сдвиг
        output = self.gamma * self.x_normalized + self.beta

        return output

    def backward(self, dout):
        """
        Обратный проход для LayerNormalization.

        Args:
            dout (np.ndarray): Градиент потерь по отношению к выходу слоя.

        Returns:
            np.ndarray: Градиент потерь по отношению ко входу слоя (dx).
        """
        # Градиенты для обучаемых параметров
        self.dgamma = np.sum(dout * self.x_normalized, axis=tuple(range(dout.ndim - 1)))
        self.dbeta = np.sum(dout, axis=tuple(range(dout.ndim - 1)))

        # Градиент по отношению к нормализованному входу
        dx_hat = dout * self.gamma

        # Промежуточные градиенты
        dvar = np.sum(dx_hat * (self.x - self.x_mean) * -0.5 * (self.x_std**2 + self.epsilon)**(-1.5), axis=-1, keepdims=True)
        dmean = np.sum(dx_hat * -1.0 / (self.x_std + self.epsilon), axis=-1, keepdims=True) + \
                dvar * np.sum(-2.0 * (self.x - self.x_mean), axis=-1, keepdims=True) / self.d_model

        # Финальный градиент по отношению ко входу x
        dx = (dx_hat / (self.x_std + self.epsilon)) + \
             (dvar * 2.0 * (self.x - self.x_mean) / self.d_model) + \
             (dmean / self.d_model)

        return dx
