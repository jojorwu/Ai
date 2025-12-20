import numpy as np

def cosine_decay_with_warmup(current_step, training_steps, warmup_steps, max_lr, min_lr):
    """
    Вычисляет learning rate на основе косинусного спада с предварительным прогревом.

    Args:
        current_step (int): Текущий шаг обучения.
        training_steps (int): Общее количество шагов обучения.
        warmup_steps (int): Количество шагов для "прогрева".
        max_lr (float): Максимальный (базовый) learning rate.
        min_lr (float): Минимальный learning rate в конце спада.

    Returns:
        float: Вычисленный learning rate для текущего шага.
    """
    if current_step < warmup_steps:
        # Линейный прогрев
        return max_lr * (current_step + 1) / warmup_steps

    if current_step > training_steps:
        # Если обучение продолжается дольше запланированного, используем min_lr
        return min_lr

    # Косинусный спад
    decay_ratio = (current_step - warmup_steps) / (training_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)
