"""
Module for loading multimodal data (text, images).
"""
from typing import List, Optional, Tuple
import torch

from src.data.multimodal_loader import MultimodalDataLoader
from src.data.batcher import Batcher


def load_multimodal_data_from_directory(
        directory_path: str) -> List[Tuple[str, Optional[torch.Tensor]]]:
    """
    Scans a directory, finds text-image pairs, and loads them in parallel.
    (Backward compatibility wrapper)
    """
    loader = MultimodalDataLoader()
    return loader.load_from_directory(directory_path)


def get_batches_torch(data, batch_size, seq_len, device):
    """
    Generator function to yield batches of data as PyTorch tensors.
    (Backward compatibility wrapper)
    """
    return Batcher.get_batches(data, batch_size, seq_len, device)
