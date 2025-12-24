"""
Реализация Dropout для регуляризации.
"""
from backend import np

class Dropout:
    """
    Слой Dropout. Во время обучения случайным образом обнуляет часть
    входных элементов с вероятностью `rate`, чтобы предотвратить переобучение.
    Во время оценки (evaluation) слой ничего не меняет.
    """
    def __init__(self, rate=0.1):
        """
        Инициализация.
        Args:
            rate (float): Вероятность обнуления элемента.
        """
        self.rate = rate
        self.mask = None
        self.is_training = True

    def train(self):
        """Включает режим обучения."""
        self.is_training = True

    def eval(self):
        """Включает режим оценки."""
        self.is_training = False

    def forward(self, x):
        """
        Прямой проход Dropout.
        """
        if self.is_training:
            self.mask = np.random.binomial(1, 1.0 - self.rate, size=x.shape) / (1.0 - self.rate)
            return x * self.mask
        return x

    def backward(self, d_out):
        """
        Обратный проход для Dropout.
        """
        if self.mask is None or not self.is_training:
            return d_out
        return d_out * self.mask
