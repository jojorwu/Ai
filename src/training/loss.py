"""
Custom loss functions for the Transformer model.
"""
import torch
from torch.nn import functional as F


def cross_entropy_with_label_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float,
    ignore_index: int = -100,  # Standard ignore_index for PyTorch loss functions
) -> torch.Tensor:
    """
    Calculates cross-entropy loss with label smoothing.

    This function penalizes the model for being too confident in its predictions,
    which can help with generalization. Instead of requiring the model to predict
    the target token with 100% probability (confidence of 1.0), it encourages a
    slightly softer distribution.

    Args:
        logits: The raw, unnormalized output from the model.
                Shape: (batch_size, seq_len, vocab_size)
        targets: The ground truth token indices.
                 Shape: (batch_size, seq_len)
        smoothing: The label smoothing factor. A value of 0.0 means no smoothing.
        vocab_size: The total number of tokens in the vocabulary.
        ignore_index: Specifies a target value that is ignored and does not
                      contribute to the input gradient.

    Returns:
        The calculated loss as a single scalar tensor.
    """
    # Reshape for cross-entropy calculation
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)

    # Use PyTorch's built-in cross_entropy with label_smoothing for efficiency.
    # This avoids creating a large smoothed_targets tensor and is more stable.
    return F.cross_entropy(
        logits_flat,
        targets_flat,
        label_smoothing=smoothing,
        ignore_index=ignore_index,
        reduction="mean",
    )
