"""
This module contains utility functions for the neural network components.
"""
import numpy as np


def softmax(logits):
    """Numerically stable Softmax function."""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def log_softmax(logits):
    """Numerically stable log_softmax function."""
    max_logits = np.max(logits, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(logits - max_logits), axis=-1, keepdims=True))
    return logits - max_logits - log_sum_exp
