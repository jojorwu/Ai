"""
Handles batch generation for training.
"""
import torch


class Batcher:
    """
    Generates batches of data as PyTorch tensors using optimized slicing.
    """

    @staticmethod
    def get_batches(data, batch_size, seq_len, device):
        """
        Generator function to yield batches of data.
        """
        # Ensure data is a torch.Tensor on the correct device
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.long, device=device)
        else:
            data = data.to(device)

        num_sequences = len(data) - seq_len
        if num_sequences <= 0:
            return

        # Pre-calculate sequence offsets for vectorized indexing
        seq_offsets = torch.arange(seq_len, device=device)

        for i in range(0, num_sequences, batch_size):
            batch_end = min(i + batch_size, num_sequences)

            # Vectorized generation of indices for the current batch
            starts = torch.arange(i, batch_end, device=device).unsqueeze(1)
            indices = starts + seq_offsets # Shape: (batch_actual, seq_len)

            x = data[indices]
            y = data[indices + 1]
            yield x, y, None # Placeholder for image data
