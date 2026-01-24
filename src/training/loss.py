"""
Custom loss functions for the Transformer model.
"""
import torch
from torch.nn import functional as F


def cross_entropy_with_label_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float,
    vocab_size: int,
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
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    # Create the smoothed target distribution
    # The confidence for the true class is (1.0 - smoothing)
    # The confidence for all other classes is (smoothing / (vocab_size - 1))
    confidence = 1.0 - smoothing
    low_confidence = smoothing / (vocab_size - 1)

    # Create a tensor of the same shape as logits, filled with the low_confidence value
    smoothed_targets = torch.full_like(
        logits_flat, low_confidence, device=logits_flat.device
    )

    # Set the high confidence value for the true target classes
    smoothed_targets.scatter_(1, targets_flat.unsqueeze(1), confidence)

    # Apply ignore_index mask
    mask = targets_flat != ignore_index
    smoothed_targets[targets_flat == ignore_index] = 0

    # Calculate the Kullback-Leibler divergence loss
    # Using log_softmax is more numerically stable than softmax followed by log
    log_probs = F.log_softmax(logits_flat, dim=-1)
    loss = torch.sum(-smoothed_targets * log_probs, dim=-1)

    # Apply the mask and calculate the mean loss for non-ignored tokens
    masked_loss = loss * mask
    final_loss = masked_loss.sum() / mask.sum()

    return final_loss
