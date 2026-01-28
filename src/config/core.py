"""
Core Pydantic models for project configuration.
"""
import argparse
import json
import logging
from typing import Any, Optional, Type, Union

from pydantic import BaseModel, Field

from src.config.hardware_config import HardwareConfig
from src.config.model_config import ComplexityConfig, ModelConfig, VisionConfig
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
                            description="Начальный текст для генерации.")
    max_len: int = Field(...,
                         description="Максимальная длина сгенерированного текста.")
    temperature: float = Field(..., description="Температура для сэмплирования.")
    top_k: int = Field(..., description="Параметр Top-k для сэмплирования.")
    top_p: float = Field(..., description="Параметр Top-p (nucleus) для сэмплирования.")
    speculative_steps: int = Field(...,
                                   description="Количество шагов спекулятивного декодирования.")
    value_threshold: float = Field(
        ..., description="Порог функции ценности для принятия спекулятивной генерации.")
    max_thought_len: int = Field(...,
                                 description="Максимальная длина 'мыслей'.")
    max_retries: int = Field(
        ..., description="Максимальное количество повторов при неудачной спекуляции.")
    max_turns: int = Field(
        10, description="Максимальное количество итераций в цикле агента.")
    context_window_size: int = Field(
        2048, description="Количество токенов, удерживаемых в истории.")


class BaseConfig(BaseModel):
    """Base configuration model with shared settings."""
    model: ModelConfig = Field(..., description="Настройки архитектуры Transformer.")
    vision: VisionConfig = Field(..., description="Настройки визуального энкодера.")
    ltm: LTMConfig = Field(..., description="Настройки Long-Term Memory (памяти агента).")
    hardware: HardwareConfig = Field(..., description="Настройки оборудования и ускорения.")
    complexity: Optional[ComplexityConfig] = Field(
        default_factory=ComplexityConfig, description="Настройки управления сложностью."
    )

    @classmethod
    def from_json(cls, file_path: str):
        """Loads and validates the configuration from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.model_validate(data)

    @staticmethod
    def print_explanations(config_class: Type['BaseConfig']):
        """Prints Russian explanations for all fields in the configuration classes."""
        print(f"\n=== Описание настроек ({config_class.__name__}) ===\n")

        def _print_recursive(model_cls, indent=0):
            for name, field in model_cls.model_fields.items():
                desc = field.description or "Нет описания."
                print("  " * indent + f"• {name}: {desc}")

                # Safely extract the underlying model class if it's nested or Optional
                target_cls = field.annotation
                actual_cls = None

                # Handle Union (e.g., Optional[Model])
                if hasattr(target_cls, "__origin__") and target_cls.__origin__ is Union:
                    for arg in target_cls.__args__:
                        if hasattr(arg, 'model_fields') and arg is not BaseModel:
                            actual_cls = arg
                            break
                elif hasattr(target_cls, 'model_fields') and target_cls is not BaseModel:
                    actual_cls = target_cls

                if actual_cls:
                    _print_recursive(actual_cls, indent + 2)

        _print_recursive(config_class)
        print("\n" + "=" * 50 + "\n")

    def to_transformer_config(self) -> TransformerConfig:
        """Converts the base config to a TransformerConfig."""
        return TransformerConfig(
            vocab_size=self.model.vocab_size or 0,
            model=self.model,
            vision=self.vision,
            ltm=self.ltm,
        )

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
        if hasattr(args, 'num_threads') and args.num_threads is not None:
            self.hardware.num_threads = args.num_threads
            logging.info("Setting num_threads to %d.", args.num_threads)
        if hasattr(args, 'num_interop_threads') and args.num_interop_threads is not None:
            self.hardware.num_interop_threads = args.num_interop_threads
            logging.info("Setting num_interop_threads to %d.", args.num_interop_threads)
        if hasattr(args, 'disable_mkldnn') and args.disable_mkldnn:
            self.hardware.enable_mkldnn = False
            logging.info("Disabling MKLDNN optimizations.")
        if hasattr(args, 'flush_denormals') and args.flush_denormals:
            self.hardware.flush_denormals = True
            logging.info("Enabling flushing denormals.")
        if hasattr(args, 'num_workers') and args.num_workers is not None:
            self.hardware.num_workers = args.num_workers
            logging.info("Setting num_workers to %d.", args.num_workers)
        if hasattr(args, 'pin_memory') and args.pin_memory is not None:
            self.hardware.pin_memory = args.pin_memory
            logging.info("Setting pin_memory to %s.", args.pin_memory)


class TrainConfig(BaseConfig):
    """Configuration model for training."""
    evolution: EvolutionConfig = Field(..., description="Параметры эволюционного обучения.")
    optimizer: OptimizerConfig = Field(..., description="Параметры оптимизатора AdamW.")
    scheduler: SchedulerConfig = Field(..., description="Параметры планировщика скорости обучения.")


class GenerateConfig(BaseConfig):
    """Configuration model for generation."""
    generation: DynamicParametersConfig
