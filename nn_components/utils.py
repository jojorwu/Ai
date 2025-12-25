"""
Shared utility functions for neural network components.
"""
from backend import np


def softmax(x):
    """
    Numerically stable softmax function.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def log_softmax(x):
    """
    Numerically stable log_softmax function.
    """
    max_x = np.max(x, axis=-1, keepdims=True)
    log_sum_exp = max_x + np.log(np.sum(np.exp(x - max_x), axis=-1, keepdims=True))
    return x - log_sum_exp
