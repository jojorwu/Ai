import numpy as np

class RMSNorm:
    """
    Реализация Root Mean Square Normalization.
    """
    def __init__(self, d_model, epsilon=1e-5):
        self.d_model = d_model
        self.epsilon = epsilon
        # Обучаемый параметр только gamma (масштаб)
        self.gamma = np.ones(d_model)

        # Кеш для обратного прохода
        self.x = None
        self.rms = None
        self.dgamma = None

    def get_params(self):
        """Возвращает сам объект слоя."""
        return [self]

    def get_trainable_params(self):
        """Возвращает словарь с обучаемыми параметрами и их градиентами."""
        return {'gamma': (self.gamma, self.dgamma)}

    def forward(self, x):
        """
        Прямой проход для RMSNorm.
        y = (x / sqrt(mean(x^2) + eps)) * gamma
        """
        self.x = x
        # Вычисляем корень из среднего квадратов по последней оси
        self.rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + self.epsilon)

        # Нормализуем и применяем масштабирование
        normalized_x = x / self.rms
        output = self.gamma * normalized_x

        return output

    def backward(self, dout):
        """
        Обратный проход для RMSNorm.
        """
        normalized_x = self.x / self.rms

        # 1. Градиент по gamma
        self.dgamma = np.sum(dout * normalized_x, axis=tuple(range(dout.ndim - 1)))

        # 2. Градиент по нормализованному входу
        d_normalized_x = dout * self.gamma

        # 3. Градиент по rms
        # dL/drms = sum(dL/dy * dy/dnorm * dnorm/drms) = sum(d_normalized_x * (-x / rms^2))
        d_rms = -np.sum(d_normalized_x * self.x, axis=-1, keepdims=True) / (self.rms**2)

        # 4. Градиент по x
        # dL/dx = dL/dnorm * dnorm/dx + dL/drms * drms/dx
        # dnorm/dx = 1/rms
        # drms/dx = (1/(2*rms)) * (2x/N) = x / (N*rms)
        dx = (d_normalized_x / self.rms) + (d_rms * self.x / (self.d_model * self.rms))

        return dx
