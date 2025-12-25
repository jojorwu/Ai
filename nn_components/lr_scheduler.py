"""
This module implements the learning rate scheduler.
"""
import numpy as np


def cosine_decay_with_warmup(current_step, training_steps, warmup_steps, max_lr, min_lr):
    """
    Calculates the learning rate based on cosine decay with a warm-up phase.

    Args:
        current_step (int): The current training step.
        training_steps (int): The total number of training steps.
        warmup_steps (int): The number of steps for the "warm-up" phase.
        max_lr (float): The maximum (base) learning rate.
        min_lr (float): The minimum learning rate at the end of the decay.

    Returns:
        float: The calculated learning rate for the current step.
    """
    if current_step < warmup_steps:
        # Linear warm-up
        return max_lr * (current_step + 1) / warmup_steps

    if current_step > training_steps:
        # If training continues longer than planned, use min_lr
        return min_lr

    # Cosine decay
    decay_ratio = (current_step - warmup_steps) / (training_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)
