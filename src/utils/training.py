"""
Common training and gradient-related utilities.
"""
from typing import Iterable
import torch
from torch import nn


def calculate_gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    """
    Calculates the Frobenius norm of the gradients for the given parameters.
    Optimized to avoid large temporary tensor concatenation.
    """
    grad_tensors = [
        p.grad.detach()
        for p in parameters
        if p.grad is not None
    ]

    if not grad_tensors:
        return 0.0

    # ||[a, b]|| = sqrt(||a||^2 + ||b||^2)
    total_norm_sq = sum(t.pow(2).sum() for t in grad_tensors)
    return torch.sqrt(total_norm_sq).item()
