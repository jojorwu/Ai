"""
Implementation of a learning rate scheduler.
"""
from backend import np


def cosine_decay_with_warmup(step, total_steps, max_lr, min_lr, warmup_steps):
    """
    Computes the learning rate with a cosine decay schedule and a linear warmup phase.
    """
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
