"""
Manages device placement for the Transformer model.
"""
import torch
from src.config import HardwareConfig


class DeviceManager:
    """
    Manages device placement for the Transformer model based on hardware
    availability and configuration.
    """

    def __init__(self, config: HardwareConfig):
        self.config = config
        self.cuda_available = torch.cuda.is_available()
        self.mps_available = torch.backends.mps.is_available()

    def get_device_map(self) -> str | dict:
        """
        Determines the appropriate device map for `accelerate`.

        Returns:
            A string ("auto") or a dictionary representing the device map.
        """
        if self.config.strategy == "hybrid" and self.cuda_available:
            # Hybrid strategy: LTM on CPU, rest on GPU
            return {"layers.long_term_memory": "cpu", "": "cuda:0"}

        # For all other cases (CPU-only, GPU-only, ARM), let accelerate decide.
        return "auto"

    def should_disable_4bit(self) -> bool:
        """
        Determines if 4-bit quantization should be disabled.
        4-bit is only supported on CUDA.
        """
        return not self.cuda_available
