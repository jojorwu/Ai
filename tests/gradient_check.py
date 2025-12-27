"""
This module provides functions for numerical gradient checking.
"""
import logging

from backend import np


# pylint: disable=too-many-arguments
def check_gradient(test_case, analytical_grad, numerical_grad, name, atol=1e-4, rtol=1e-5):
    """
    Checks if the analytical and numerical gradients are close.
    """
    is_close = np.allclose(analytical_grad, numerical_grad, atol=atol, rtol=rtol)
    if not is_close:
        logging.error("Gradient check for %s FAILED", name)
        logging.error("Analytical grad: %s", analytical_grad)
        logging.error("Numerical grad: %s", numerical_grad)
        logging.error("Difference: %s", analytical_grad - numerical_grad)
    test_case.assertTrue(is_close, f"Gradient check for {name} FAILED")


def numerical_gradient(model_forward, weights, dout, epsilon=1e-5):
    """
    Computes the numerical gradient of a model's forward pass.
    """
    grad = np.zeros_like(weights)
    it = np.nditer(weights, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_value = weights[idx]

        weights[idx] = old_value + epsilon
        pos_loss = np.sum(model_forward(weights) * dout)

        weights[idx] = old_value - epsilon
        neg_loss = np.sum(model_forward(weights) * dout)

        grad[idx] = (pos_loss - neg_loss) / (2 * epsilon)
        weights[idx] = old_value
        it.iternext()
    return grad
