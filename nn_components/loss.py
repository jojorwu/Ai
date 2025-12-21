import numpy as np
from nn_components.utils import log_softmax

class SoftmaxCrossEntropy:
    """
    Комбинированный слой Softmax + Cross-Entropy Loss, использующий
    численно стабильный log_softmax для вычислений.
    """
    def __init__(self):
        self.probs = None
        self.targets = None

    def forward(self, logits, targets):
        """
        Прямой проход для вычисления потерь через NLL Loss и log_softmax.

        Args:
            logits (np.ndarray): Выход модели (размер: batch_size, seq_len, vocab_size).
            targets (np.ndarray): Целевые индексы (размер: batch_size, seq_len).

        Returns:
            float: Среднее значение потерь.
        """
        batch_size, seq_len, vocab_size = logits.shape

        # 1. Вычисляем численно стабильные логарифмы вероятностей
        log_probs = log_softmax(logits)

        # Сохраняем вероятности и цели для обратного прохода
        # Вероятности понадобятся для вычисления градиента
        self.probs = np.exp(log_probs)
        self.targets = targets

        # 2. Выбираем логарифмы вероятностей для правильных классов
        batch_indices = np.arange(batch_size)[:, np.newaxis]
        seq_indices = np.arange(seq_len)
        correct_class_log_probs = log_probs[batch_indices, seq_indices, targets]

        # 3. Вычисляем Negative Log Likelihood Loss
        loss = -np.sum(correct_class_log_probs)

        # 4. Усредняем потери
        loss /= (batch_size * seq_len)

        return loss

    def backward(self):
        """
        Обратный проход для вычисления градиента по отношению к логитам.
        Градиент для log_softmax + NLLLoss такой же, как и для softmax + CrossEntropy: (probs - y_one_hot).

        Returns:
            np.ndarray: Градиент dL/dZ (размер: batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len, vocab_size = self.probs.shape

        # Градиент - это (P - Y), где P - вероятности, Y - one-hot цели
        dx = self.probs.copy()

        # Вычитаем 1 из вероятностей для правильных классов (создаем Y)
        batch_indices = np.arange(batch_size)[:, np.newaxis]
        seq_indices = np.arange(seq_len)
        dx[batch_indices, seq_indices, self.targets] -= 1

        # Нормализуем градиент так же, как и потери
        dx /= (batch_size * seq_len)

        return dx

class MSELoss:
    """Mean Squared Error Loss."""
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true)**2)

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.size

class MarginRankingLoss:
    """Margin Ranking Loss."""
    def __init__(self, margin=1.0):
        self.margin = margin

    def forward(self, y_good, y_bad):
        self.y_good = y_good
        self.y_bad = y_bad
        self.loss = np.maximum(0, self.margin - (y_good - y_bad))
        return np.mean(self.loss)

    def backward(self):
        # Gradient is -1 for the "good" input and +1 for the "bad" input if the margin is not met
        mask = (self.loss > 0).astype(int)
        d_y_good = -mask / self.y_good.size
        d_y_bad = mask / self.y_bad.size
        return d_y_good, d_y_bad
