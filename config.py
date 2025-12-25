"""
Pydantic models for strong typing and validation of the project configuration.
"""
import json
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from nn_components.kv_cache import KVCache


class ModelConfig(BaseModel):
    """Configuration for the Transformer model architecture."""
    d_model: int = Field(..., description="Dimensionality of the model's vectors.")
    num_layers: int = Field(..., description="Number of layers in the encoder and decoder.")
    num_heads: int = Field(..., description="Number of heads in Multi-Head Attention.")
    num_kv_heads: int = Field(
        ..., description="Number of heads for Key/Value in Grouped-Query Attention."
    )
    d_ff: int = Field(..., description="Dimensionality in Feed-Forward layers.")
    max_seq_len: int = Field(..., description="Maximum sequence length.")
    dropout_rate: float = Field(..., description="Dropout probability.")
    ltm_d_hidden: Optional[int] = Field(
        None, description="Dimensionality of the hidden layer in LTM."
    )
    ltm_num_layers: Optional[int] = Field(None, description="Number of layers in LTM.")
    num_experts: Optional[int] = Field(None, description="Number of 'experts' in the MoE layer.")
    top_k_experts: Optional[int] = Field(
        None, description="Number of 'experts' to select for each token."
    )


class DecoderBlockConfig(BaseModel):
    """Configuration for a single DecoderBlock."""
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float
    num_kv_heads: int
    num_layers: int
    num_experts: Optional[int]
    top_k_experts: Optional[int]


class MultiHeadAttentionConfig(BaseModel):
    """Configuration for the MultiHeadAttention layer."""
    d_model: int
    num_heads: int
    num_kv_heads: int
    num_layers: int


class VisionConfig(BaseModel):
    """Configuration for the Vision Encoder."""
    image_size: tuple[int, int] = Field((224, 224), description="Input image size (height, width).")
    patch_size: int = Field(16, description="Size of a single image patch.")
    num_channels: int = Field(3, description="Number of channels in the image (e.g., 3 for RGB).")


class EvolutionConfig(BaseModel):
    """Configuration for the evolutionary training process."""
    pretrain_epochs: int = Field(
        ..., description="Number of epochs for initial pre-training of the base model."
    )
    evolution_epochs: int = Field(
        ..., description="Number of evolution cycles (generations) for agents."
    )
    num_agents: int = Field(..., description="Number of agents in a single population.")
    num_survivors: int = Field(
        ..., description="Number of best agents whose LTMs will be merged."
    )
    batch_size: int = Field(..., description="Size of a single batch.")
    seq_len: int = Field(..., description="Sequence length for training.")
    gradient_accumulation_steps: int = Field(
        ..., description="Number of steps to accumulate gradients."
    )
    validation_split: float = Field(..., description="Fraction of data to use for validation.")
    data_dir: str = Field(..., description="Directory with training data.")
    weights_path: str = Field(..., description="Path to save the final model weights.")
    checkpoint_path: Optional[str] = Field(None, description="Path to save checkpoints.")
    early_stopping_patience: int = Field(
        3, description="Number of epochs without improvement for early stopping."
    )
    best_model_path: str = Field("best_model.npz", description="Path to save the best model.")
    moe_aux_loss_coeff: float = Field(
        0.01, description="Coefficient for the MoE auxiliary 'balancing' loss."
    )


class OptimizerConfig(BaseModel):
    """Configuration for the Adam optimizer."""
    learning_rate: float = Field(..., description="Learning rate.")
    beta1: float = Field(..., description="Beta1 coefficient for Adam.")
    beta2: float = Field(..., description="Beta2 coefficient for Adam.")
    epsilon: float = Field(..., description="Small value to prevent division by zero.")
    weight_decay: float = Field(..., description="Weight decay coefficient (L2 regularization).")
    max_norm: float = Field(..., description="Maximum norm for gradient clipping.")


class LTMOptimizerConfig(BaseModel):
    """Configuration for the LTM's Adam optimizer."""
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float


class LTMConfig(BaseModel):
    """Configuration for the Long-Term Memory (LTM)."""
    surprise_threshold: float = Field(..., description="'Surprise' threshold for updating the LTM.")
    optimizer: LTMOptimizerConfig


class SchedulerConfig(BaseModel):
    """Configuration for the learning rate scheduler."""
    warmup_steps: int = Field(..., description="Number of 'warm-up' steps.")
    min_lr: float = Field(..., description="Minimum learning rate value.")


class GenerationConfig(BaseModel):
    """Configuration for the text generation process."""
    start_text: str = Field(..., description="Initial text for generation.")
    max_len: int = Field(..., description="Maximum length of the generated text.")
    temperature: float = Field(..., description="Temperature for sampling.")
    top_k: int = Field(..., description="Top-k for sampling.")
    top_p: float = Field(..., description="Top-p (nucleus) for sampling.")
    speculative_steps: int = Field(..., description="Number of speculative steps.")
    value_threshold: float = Field(
        ..., description="Value threshold for accepting speculative generation."
    )
    max_thought_len: int = Field(..., description="Maximum length of 'thoughts'.")
    max_retries: int = Field(..., description="Maximum number of retries on failed speculation.")
    max_turns: int = Field(10, description="Maximum number of iterations (tool calls) in the agent loop.")
    context_window_size: int = Field(
        2048, description="The number of tokens to retain in the conversation history."
    )


class HardwareConfig(BaseModel):
    """Hardware configuration."""
    device: Literal["cpu", "gpu", "mps"] = Field(
        "cpu", description="Device for computations (cpu, gpu, mps)."
    )


@dataclass
class ForwardPassInput:
    """Dataclass for inputs to a DecoderBlock's forward pass."""
    x: np.ndarray
    ltm_state: np.ndarray
    mask: Optional[np.ndarray]
    kv_cache: Optional[KVCache]
    layer_idx: int
    seq_offset: int


class Config(BaseModel):
    """Main configuration model."""
    model: ModelConfig
    vision: VisionConfig
    evolution: EvolutionConfig
    optimizer: OptimizerConfig
    ltm: LTMConfig
    scheduler: SchedulerConfig
    generation: GenerationConfig
    hardware: HardwareConfig

    @classmethod
    def from_json(cls, file_path: str) -> 'Config':
        """Loads and validates the configuration from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
