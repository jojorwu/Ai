"""
Module containing the loss layer.
"""
import numpy as np

from src.nn_components.utils import log_softmax


class SoftmaxCrossEntropy:
    """
    A combined Softmax + Cross-Entropy Loss layer that uses a numerically stable
    log_softmax for calculations.
    """
    def __init__(self):
        self.probs = None
        self.targets = None
        self.reduction = 'mean'

    def forward(self, logits, targets, reduction='mean'):
        """
        Forward pass for loss calculation.
        Supports 'mean' and 'none' for reduction.
        """
        self.reduction = reduction
        batch_size, seq_len, _ = logits.shape

        log_probs = log_softmax(logits)
        self.probs = np.exp(log_probs)
        self.targets = targets

        batch_indices = np.arange(batch_size)[:, np.newaxis]
        seq_indices = np.arange(seq_len)
        correct_class_log_probs = log_probs[batch_indices, seq_indices, targets]

        # Negative Log Likelihood Loss
        loss = -correct_class_log_probs

        if reduction == 'mean':
            return np.mean(loss)
        if reduction == 'none':
            # Возвращаем средние потери для каждой последовательности в батче
            return np.mean(loss, axis=1)

        raise ValueError("Неподдерживаемый тип reduction. Используйте 'mean' или 'none'.")


    def backward(self):
        """
        Backward pass to compute the gradient with respect to the logits.
        """
        batch_size, seq_len, _ = self.probs.shape

        dx = self.probs.copy()
        batch_indices = np.arange(batch_size)[:, np.newaxis]
        seq_indices = np.arange(seq_len)
        dx[batch_indices, seq_indices, self.targets] -= 1

        # Нормализуем градиент в соответствии с reduction
        if self.reduction == 'mean':
            dx /= (batch_size * seq_len)
        elif self.reduction == 'none':
            # Если потерь не было, градиент должен быть усреднен по batch_size
            dx /= seq_len

        return dx

class MarginRankingLoss:
    """Implements the Margin Ranking Loss."""
    def __init__(self, margin=1.0):
        """
        Initializes the MarginRankingLoss.
        Args:
            margin (float): The margin for the loss.
        """
        self.margin = margin
        self.y_good = None
        self.y_bad = None
        self.loss = None

    def forward(self, y_good, y_bad):
        """Forward pass for the Margin Ranking Loss."""
        self.y_good = y_good
        self.y_bad = y_bad
        self.loss = np.maximum(0, self.margin - (y_good - y_bad))
        return np.mean(self.loss)

    def backward(self):
        """Backward pass for the Margin Ranking Loss."""
        # Gradient is -1 for the "good" input and +1 for the "bad" input if the margin is not met
        mask = (self.loss > 0).astype(int)
        d_y_good = -mask / self.y_good.size
        d_y_bad = mask / self.y_bad.size
        return d_y_good, d_y_bad
