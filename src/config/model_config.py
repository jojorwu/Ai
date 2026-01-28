"""
Pydantic models for model architecture and vision configuration.
"""
from typing import Any, Literal, Optional, Tuple
from pydantic import BaseModel, Field


class MultiHeadAttentionConfig(BaseModel):
    """Configuration for the Multi-Head Attention layer."""
    d_model: int
    num_heads: int
    num_kv_heads: int
    rotary_emb: Optional[Tuple[Any, Any]] = None
    bias: bool = False
    num_layers: int = 1


class MoEConfig(BaseModel):
    """Configuration for the Mixture of Experts layer."""
    d_model: int
    d_ff: int
    num_experts: int
    top_k: int
    bias: bool = False


class FeedForwardConfig(BaseModel):
    """Configuration for the Feed-Forward Network layer."""
    d_model: int
    d_ff: int
    bias: bool = False
    num_layers: int = 1


class DecoderBlockConfig(BaseModel):
    """Configuration for a single DecoderBlock."""
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float
    num_kv_heads: int
    num_layers: int
    num_experts: Optional[int] = None
    top_k_experts: Optional[int] = None
    rotary_emb: Optional[Tuple[Any, Any]] = None
    long_term_memory: Optional[Any] = None
    load_in_4bit: bool = False
    class Config:  # pylint: disable=too-few-public-methods
        """Pydantic config."""
        arbitrary_types_allowed = True


class LTMArchitectureConfig(BaseModel):
    """Configuration specific to the LTM architecture."""
    d_hidden: int | None = Field(
        None, description="Dimensionality of the hidden layer in LTM.")
    num_layers: int | None = Field(None,
                                       description="Number of layers in LTM.")


class ModelConfig(BaseModel):
    """Configuration for the Transformer model architecture."""
    vocab_size: int | None = Field(
        None, description="Size of the vocabulary.")
    d_model: int = Field(...,
                         description="Dimensionality of the model's vectors.")
    num_layers: int = Field(
        ..., description="Number of layers in the encoder and decoder.")
    num_heads: int = Field(...,
                           description="Number of heads in Multi-Head Attention.")
    num_kv_heads: int = Field(
        ...,
        description="Number of heads for Key/Value in Grouped-Query Attention.")
    d_ff: int = Field(..., description="Dimensionality in Feed-Forward layers.")
    max_seq_len: int = Field(..., description="Maximum sequence length.")
    dropout_rate: float = Field(..., description="Dropout probability.")
    ltm: LTMArchitectureConfig = Field(
        default_factory=LTMArchitectureConfig,
        description="Configuration for the Long-Term Memory module."
    )
    num_experts: int | None = Field(
        None, description="Number of 'experts' in the MoE layer.")
    top_k_experts: int | None = Field(
        None, description="Number of 'experts' to select for each token.")
    gradient_checkpointing: bool = Field(
        False, description="Enable gradient checkpointing to save memory.")


class VisionConfig(BaseModel):
    """Configuration for the Vision Encoder."""
    image_size: tuple[int, int] = Field(
        (224, 224), description="Input image size (height, width).")
    patch_size: int = Field(16,
                            description="Size of a single image patch.")
    num_channels: int = Field(
        3, description="Number of channels in the image (e.g., 3 for RGB).")


class ComplexityConfig(BaseModel):
    """Configuration for dynamic parameter adjustment."""
    low_complexity_threshold: float = 0.5
    medium_complexity_threshold: float = 1.5
    high_complexity_threshold: float = 3.0
    low_complexity_top_k: int = 1
    medium_complexity_top_k: int = 10
    high_complexity_top_k: int = 50
