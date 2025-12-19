import numpy as np

def softmax(logits):
    """Стабильная функция Softmax."""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
