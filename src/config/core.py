"""
Core Pydantic models for project configuration.
"""
import json
from typing import Literal

from pydantic import BaseModel, Field

from .dynamic_parameters_config import DynamicParametersConfig
from .model_config import ModelConfig, VisionConfig
from .training_config import EvolutionConfig, LTMConfig, OptimizerConfig, SchedulerConfig


class HardwareConfig(BaseModel):
    """Hardware configuration."""
    device: Literal["cpu", "gpu", "mps"] = Field(
        "cpu", description="Device for computations (cpu, gpu, mps).")
    strategy: Literal["unified", "discrete", "hybrid"] = Field(
        "discrete", description="Memory strategy for hardware.")
    torch_compile: bool = Field(
        False, description="Enable torch.compile for the model.")


class BaseConfig(BaseModel):
    """Base configuration model with shared settings."""
    model: ModelConfig
    vision: VisionConfig
    ltm: LTMConfig
    hardware: HardwareConfig

    @classmethod
    def from_json(cls, file_path: str):
        """Loads and validates the configuration from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.model_validate(data)


class TrainConfig(BaseConfig):
    """Configuration model for training."""
    evolution: EvolutionConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig


class GenerateConfig(BaseConfig):
    """Configuration model for generation."""
    generation: DynamicParametersConfig
