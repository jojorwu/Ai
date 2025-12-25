"""
This module contains the implementation of loss functions used in the model.
"""
import numpy as np

from nn_components.utils import log_softmax


class SoftmaxCrossEntropy:
    """
    Combined Softmax + Cross-Entropy Loss layer that uses a numerically
    stable log_softmax for computation.
    """

    def __init__(self):
        self.probs = None
        self.targets = None
        self.reduction = 'mean'

    def forward(self, logits, targets, reduction='mean'):
        """
        Forward pass for computing the loss.
        Supports 'mean' and 'none' for reduction.
        """
        self.reduction = reduction
        batch_size, seq_len, _ = logits.shape

        log_probs = log_softmax(logits)
        self.probs = np.exp(log_probs)
        self.targets = targets

        batch_indices = np.arange(batch_size)[:, np.newaxis]
        seq_indices = np.arange(seq_len)
        correct_class_log_probs = log_probs[
            batch_indices, seq_indices, targets]

        # Negative Log Likelihood Loss
        loss = -correct_class_log_probs

        if reduction == 'mean':
            return np.mean(loss)
        if reduction == 'none':
            # Return the mean loss for each sequence in the batch
            return np.mean(loss, axis=1)

        raise ValueError(
            "Unsupported reduction type. Use 'mean' or 'none'.")

    def backward(self):
        """
        Backward pass for computing the gradient with respect to the logits.
        """
        batch_size, seq_len, _ = self.probs.shape

        dx = self.probs.copy()
        batch_indices = np.arange(batch_size)[:, np.newaxis]
        seq_indices = np.arange(seq_len)
        dx[batch_indices, seq_indices, self.targets] -= 1

        # Normalize the gradient according to the reduction type
        if self.reduction == 'mean':
            dx /= (batch_size * seq_len)
        elif self.reduction == 'none':
            # If reduction was 'none', the gradient is averaged per sequence
            dx /= seq_len

        return dx
