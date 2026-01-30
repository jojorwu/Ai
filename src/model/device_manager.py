"""
Manages device placement for the Transformer model.
"""
import torch
from src.config.hardware_config import HardwareConfig


class DeviceManager:
    """
    Manages device placement for the Transformer model based on hardware
    availability and configuration.
    """

    def __init__(self, config: HardwareConfig) -> None:
        self.config = config
        self.cuda_available = torch.cuda.is_available()
        self.mps_available = torch.backends.mps.is_available()

    def optimize_environment(self) -> None:
        """Applies hardware-specific optimizations."""
        if self.config.device == "cpu":
            if self.config.num_threads > 0:
                torch.set_num_threads(self.config.num_threads)
            if self.config.num_interop_threads > 0:
                torch.set_num_interop_threads(self.config.num_interop_threads)

            torch.backends.mkldnn.enabled = self.config.enable_mkldnn

            if self.config.flush_denormals:
                torch.set_flush_denormal(True)

            # Performance: Preferred BFloat16 for Arm/CPU if supported.
            if hasattr(torch, "cpu") and hasattr(torch.cpu, "is_bf16_supported"):
                 # This is just an environment-level hint/check
                 pass

        elif self.config.device == "gpu" or self.cuda_available:
            torch.backends.cudnn.benchmark = True
            # Allows for some more parallelism in CUDA operations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        elif self.config.device == "mps" or self.mps_available:
            # Apple Silicon optimizations
            import os
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    def get_device_map(self) -> dict:
        """
        Determines the appropriate device map for `accelerate`.

        Returns:
            A dictionary representing the device map.
        """
        if self.config.strategy == "hybrid" and self.cuda_available:
            # Hybrid strategy: LTM on CPU, rest on GPU
            return {"layers.long_term_memory": "cpu", "": "cuda:0"}

        # Default to the configured device for other strategies
        return {"": self.config.device}

    def should_disable_4bit(self) -> bool:
        """
        Determines if 4-bit quantization should be disabled.
        4-bit is only supported on CUDA.
        """
        return not self.cuda_available
