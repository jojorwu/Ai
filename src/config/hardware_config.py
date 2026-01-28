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
