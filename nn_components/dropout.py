import numpy as np

class Dropout:
    """
    Слой Dropout для регуляризации.
    """
    def __init__(self, dropout_rate):
        """
        Инициализация слоя.

        Args:
            dropout_rate (float): Вероятность обнуления нейрона (от 0 до 1).
        """
        self.dropout_rate = dropout_rate
        self.is_training = True  # По умолчанию слой в режиме обучения
        self.mask = None

    def forward(self, x):
        """
        Прямой проход.

        Args:
            x (np.ndarray): Входной тензор.

        Returns:
            np.ndarray: Выходной тензор.
        """
        if self.is_training:
            # Во время обучения создаем маску и применяем ее
            # Используем инвертированный dropout: масштабируем на этапе обучения,
            # чтобы ничего не делать на этапе генерации.
            self.mask = (np.random.rand(*x.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            return x * self.mask
        else:
            # Во время генерации просто возвращаем вход
            return x

    def backward(self, dout):
        """
        Обратный проход.

        Args:
            dout (np.ndarray): Градиент с предыдущего слоя.

        Returns:
            np.ndarray: Градиент, пропущенный через dropout-маску.
        """
        # Градиент проходит только через те нейроны, которые были активны
        return dout * self.mask

    def get_params(self):
        # Dropout не имеет обучаемых параметров
        return []
