"""
Utilities for numerical gradient checking in tests.
"""
import logging

import numpy as np


def check_gradient(test_case, analytical_grad, numerical_grad, name):
    """Compares analytical and numerical gradients."""
    is_close = np.allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-4)
    if not is_close:
        logging.error(f"Gradient check for {name} FAILED")
        logging.error(f"Analytical grad: {analytical_grad}")
        logging.error(f"Numerical grad: {numerical_grad}")
        logging.error(f"Difference: {np.abs(analytical_grad - numerical_grad)}")
    test_case.assertTrue(is_close, f"Gradient check for {name} FAILED")
    logging.info(f"Gradient check for {name} PASSED.")


def numerical_gradient(model_forward, param, dout, epsilon=1e-5):
    """
    Computes the numerical gradient for a parameter `param` using
    the forward pass function `model_forward`.
    """
    grad_numerical = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        original_value = param[ix]

        param[ix] = original_value + epsilon
        # Pass the perturbed parameter to the forward function
        fx_plus_h = np.sum(model_forward(param) * dout)

        param[ix] = original_value - epsilon
        # Pass the perturbed parameter to the forward function
        fx_minus_h = np.sum(model_forward(param) * dout)

        grad_numerical[ix] = (fx_plus_h - fx_minus_h) / (2 * epsilon)

        param[ix] = original_value
        it.iternext()
    return grad_numerical
