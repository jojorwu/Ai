"""
Handles batch generation for training.
"""
from typing import Any, Generator, List, Optional, Tuple, Union
import torch


class Batcher:
    """
    Generates batches of data as PyTorch tensors using optimized slicing.
    """

    @staticmethod
    def get_batches(
        data: Union[torch.Tensor, List[int], Any],
        batch_size: int,
        seq_len: int,
        device: torch.device,
        pin_memory: bool = False
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], None, None]:
        """
        Generator function to yield batches of data (x, y, image).

        Args:
            data: Input data as a tensor or list of token IDs.
            batch_size: Number of sequences per batch.
            seq_len: Length of each sequence.
            device: Target device for the batches.
            pin_memory: Whether to use pinned memory for faster CPU-to-GPU transfers.
        """
        is_cuda = device.type == "cuda" if hasattr(device, "type") else "cuda" in str(device)
        should_pin = pin_memory and is_cuda

        # Ensure data is a torch.Tensor
        if not isinstance(data, torch.Tensor):
            if should_pin:
                data = torch.tensor(data, dtype=torch.long, device="cpu").pin_memory()
            else:
                data = torch.tensor(data, dtype=torch.long, device=device)
        elif should_pin and data.device.type == "cpu":
            data = data.pin_memory()

        num_sequences = len(data) - seq_len
        if num_sequences <= 0:
            return

        # Pre-calculate sequence offsets for vectorized indexing
        seq_offsets = torch.arange(seq_len, device=device)

        # We want indices to be on the same device as the data tensor for slicing
        # If data is on CPU (pinned), indices should be on CPU.
        data_device = data.device

        for i in range(0, num_sequences, batch_size):
            batch_end = min(i + batch_size, num_sequences)

            # Vectorized generation of indices for the current batch
            starts = torch.arange(i, batch_end, device=data_device).unsqueeze(1)
            indices = starts + seq_offsets.to(data_device) # Shape: (batch_actual, seq_len)

            x = data[indices]
            y = data[indices + 1]

            # Move to target device with non_blocking if it's a different device
            if x.device != device:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

            yield x, y, None # Placeholder for image data
