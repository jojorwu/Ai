"""
Утилиты для численной проверки градиентов в тестах.
"""

import numpy as np

def check_gradient(test_case, analytical_grad, numerical_grad, name):
    """Сравнивает аналитический и численный градиенты."""
    is_close = np.allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-4)
    if not is_close:
        print(f"Gradient check for {name} FAILED")
        print("Analytical grad:", analytical_grad)
        print("Numerical grad:", numerical_grad)
        print("Difference:", np.abs(analytical_grad - numerical_grad))
    test_case.assertTrue(is_close, f"Gradient check for {name} FAILED")
    print(f"Gradient check for {name} PASSED.")


def numerical_gradient(model_forward, param, dout, epsilon=1e-5):
    """
    Вычисляет численный градиент для параметра `param` с использованием
    функции прямого прохода `model_forward`.
    """
    grad_numerical = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        original_value = param[ix]

        param[ix] = original_value + epsilon
        fx_plus_h = np.sum(model_forward() * dout)

        param[ix] = original_value - epsilon
        fx_minus_h = np.sum(model_forward() * dout)

        grad_numerical[ix] = (fx_plus_h - fx_minus_h) / (2 * epsilon)

        param[ix] = original_value
        it.iternext()
    return grad_numerical
