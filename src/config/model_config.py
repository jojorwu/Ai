"""
Pydantic models for model architecture and vision configuration.
"""
from typing import Any, Literal, Optional, Tuple
from pydantic import BaseModel, Field


class MultiHeadAttentionConfig(BaseModel):
    """Configuration for the Multi-Head Attention layer."""
    d_model: int = Field(..., description="Dimensionality of the model vectors.")
    num_heads: int = Field(..., description="Number of attention heads.")
    num_kv_heads: int = Field(
        ..., description="Number of heads for Key/Value (for GQA)."
    )
    rotary_emb: Optional[Tuple[Any, Any]] = Field(
        None, description="Precomputed RoPE embeddings."
    )
    bias: bool = Field(False, description="Whether to use bias in linear layers.")
    num_layers: int = Field(1, description="Number of attention sub-layers.")


class MoEConfig(BaseModel):
    """Configuration for the Mixture of Experts layer."""
    d_model: int = Field(..., description="Dimensionality of the model vectors.")
    d_ff: int = Field(..., description="Dimensionality of the expert FFNs.")
    num_experts: int = Field(..., description="Total number of experts.")
    top_k: int = Field(..., description="Number of experts to route each token to.")
    bias: bool = Field(False, description="Whether to use bias in expert layers.")


class FeedForwardConfig(BaseModel):
    """Configuration for the Feed-Forward Network layer."""
    d_model: int = Field(..., description="Dimensionality of the model vectors.")
    d_ff: int = Field(..., description="Dimensionality of the hidden layer.")
    bias: bool = Field(False, description="Whether to use bias in linear layers.")
    num_layers: int = Field(1, description="Number of FFN sub-layers.")


class DecoderBlockConfig(BaseModel):
    """Configuration for a single DecoderBlock."""
    d_model: int = Field(..., description="Dimensionality of the model vectors.")
    num_heads: int = Field(..., description="Number of attention heads.")
    d_ff: int = Field(..., description="Dimensionality of the FFN hidden layer.")
    dropout_rate: float = Field(..., description="Dropout probability.")
    num_kv_heads: int = Field(..., description="Number of KV heads for GQA.")
    num_layers: int = Field(..., description="Total layers in the block.")
    num_experts: Optional[int] = Field(None, description="Number of experts for MoE.")
    top_k_experts: Optional[int] = Field(
        None, description="Experts to select per token."
    )
    rotary_emb: Optional[Tuple[Any, Any]] = Field(
        None, description="RoPE embeddings tuple."
    )
    long_term_memory: Optional[Any] = Field(None, description="LTM module instance.")
    load_in_4bit: bool = Field(False, description="Whether to use 4-bit quantization.")

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
    low_complexity_threshold: float = Field(
        0.5, description="Threshold for 'low' complexity routing."
    )
    medium_complexity_threshold: float = Field(
        1.5, description="Threshold for 'medium' complexity routing."
    )
    high_complexity_threshold: float = Field(
        3.0, description="Threshold for 'high' complexity routing."
    )
    low_complexity_top_k: int = Field(
        1, description="MoE top_k for low complexity tasks."
    )
    medium_complexity_top_k: int = Field(
        10, description="MoE top_k for medium complexity tasks."
    )
    high_complexity_top_k: int = Field(
        50, description="MoE top_k for high complexity tasks."
    )
