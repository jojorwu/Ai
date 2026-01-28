"""
Pydantic models for hardware strategy configuration.
"""
from typing import Literal
from pydantic import BaseModel, Field


class HardwareConfig(BaseModel):
    """Hardware configuration."""
    device: Literal["cpu", "gpu", "mps"] = Field(
        "cpu", description="Device for computations (cpu, gpu, mps).")
    strategy: Literal["unified", "discrete", "hybrid"] = Field(
        "discrete", description="Memory strategy for hardware.")
    torch_compile: bool = Field(
        False, description="Enable torch.compile for the model.")
    num_threads: int = Field(
        0, description="Number of threads for intra-op parallelism (0 for default).")
    num_interop_threads: int = Field(
        0, description="Number of threads for inter-op parallelism (0 for default).")
    enable_mkldnn: bool = Field(
        True, description="Enable oneDNN (MKLDNN) optimizations for CPU.")
    flush_denormals: bool = Field(
        False, description="Enable flushing denormal numbers to zero on CPU.")
