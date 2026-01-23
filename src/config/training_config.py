"""
Pydantic models for training configuration.
"""
from typing import Optional

from pydantic import BaseModel, Field


class EvolutionConfig(BaseModel):
    """Configuration for the evolutionary training process."""
    pretrain_epochs: int = Field(
        ..., description="Number of epochs for initial pre-training.")
    evolution_epochs: int = Field(
        ..., description="Number of evolution cycles (generations).")
    num_agents: int = Field(...,
                           description="Number of agents in a single population.")
    num_survivors: int = Field(
        ..., description="Number of best agents whose LTMs will be merged.")
    batch_size: int = Field(..., description="Size of a single batch.")
    seq_len: int = Field(..., description="Sequence length for training.")
    gradient_accumulation_steps: int = Field(
        ..., description="Number of steps to accumulate gradients.")
    validation_split: float = Field(
        ..., description="Fraction of data to use for validation.")
    data_dir: str = Field(..., description="Directory with training data.")
    weights_path: str = Field(...,
                              description="Path to save the final model weights.")
    checkpoint_path: Optional[str] = Field(None,
                                           description="Path to save checkpoints.")
    early_stopping_patience: int = Field(
        3, description="Epochs without improvement for early stopping.")
    best_model_path: str = Field("best_model.npz",
                                 description="Path to save the best model.")
    moe_aux_loss_coeff: float = Field(
        0.01, description="Coefficient for the MoE auxiliary loss.")
    label_smoothing: float = Field(
        0.0, description="Value for label smoothing (0.0 means disabled).")


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
    surprise_threshold: float = Field(
        ..., description="'Surprise' threshold for updating the LTM.")
    optimizer: LTMOptimizerConfig


class SchedulerConfig(BaseModel):
    """Configuration for the learning rate scheduler."""
    warmup_steps: int = Field(...,
                              description="Number of 'warm-up' steps.")
    training_steps: int = Field(...,
                                description="Total number of training steps.")
    min_lr: float = Field(..., description="Minimum learning rate value.")
