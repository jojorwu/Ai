"""
Core Pydantic models for project configuration.
"""
import argparse
import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.config.hardware_config import HardwareConfig
from src.config.model_config import ModelConfig, VisionConfig
from src.config.training_config import (EvolutionConfig, LTMConfig,
                                         OptimizerConfig, SchedulerConfig)


class TransformerConfig(BaseModel):
    """Configuration for the Transformer model."""
    vocab_size: int
    model: ModelConfig
    vision: VisionConfig
    ltm: Optional[LTMConfig] = None
    tokenizer: Optional[Any] = None
    class Config:  # pylint: disable=too-few-public-methods
        """Pydantic config."""
        arbitrary_types_allowed = True


class DynamicParametersConfig(BaseModel):
    """Configuration for the dynamic parameters of the text generation process."""
    start_text: str = Field(...,
                            description="Initial text for generation.")
    max_len: int = Field(...,
                         description="Maximum length of the generated text.")
    temperature: float = Field(..., description="Temperature for sampling.")
    top_k: int = Field(..., description="Top-k for sampling.")
    top_p: float = Field(..., description="Top-p (nucleus) for sampling.")
    speculative_steps: int = Field(...,
                                   description="Number of speculative steps.")
    value_threshold: float = Field(
        ..., description="Value threshold for accepting speculative generation.")
    max_thought_len: int = Field(...,
                                 description="Maximum length of 'thoughts'.")
    max_retries: int = Field(
        ..., description="Maximum number of retries on failed speculation.")
    max_turns: int = Field(
        10, description="Maximum number of iterations in the agent loop.")
    context_window_size: int = Field(
        2048, description="The number of tokens to retain in history.")


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

    def apply_cli_args(self, args: argparse.Namespace):
        """
        Overrides configuration fields based on command-line arguments.
        Args:
            args: Parsed arguments from argparse.
        """
        if args.hardware_strategy:
            self.hardware.strategy = args.hardware_strategy
            logging.info(
                "Overriding hardware strategy with '%s'", args.hardware_strategy
            )
        if args.torch_compile:
            self.hardware.torch_compile = True
            logging.info("Enabling torch.compile.")


class TrainConfig(BaseConfig):
    """Configuration model for training."""
    evolution: EvolutionConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig


class GenerateConfig(BaseConfig):
    """Configuration model for generation."""
    generation: DynamicParametersConfig
