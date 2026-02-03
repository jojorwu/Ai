"""
Pydantic models for model architecture and vision configuration.
"""
from typing import Any, Literal, Optional, Tuple
from pydantic import BaseModel, Field


class MultiHeadAttentionConfig(BaseModel):
    """Configuration for the Multi-Head Attention layer."""
    d_model: int = Field(..., description="Размерность векторов модели.")
    num_heads: int = Field(..., description="Количество голов внимания.")
    num_kv_heads: int = Field(
        ..., description="Количество голов для Key/Value (для GQA)."
    )
    rotary_emb: Optional[Tuple[Any, Any]] = Field(
        None, description="Предварительно вычисленные эмбеддинги RoPE."
    )
    bias: bool = Field(False, description="Использовать ли смещение в линейных слоях.")
    num_layers: int = Field(1, description="Количество подслоев внимания.")


class MoEConfig(BaseModel):
    """Configuration for the Mixture of Experts layer."""
    d_model: int = Field(..., description="Размерность векторов модели.")
    d_ff: int = Field(..., description="Размерность экспертных FFN.")
    num_experts: int = Field(..., description="Общее количество экспертов.")
    top_k: int = Field(..., description="Количество экспертов для каждого токена.")
    bias: bool = Field(False, description="Использовать ли смещение в слоях экспертов.")
    use_shared_expert: bool = Field(
        False, description="Использовать ли общий эксперт (shared expert)."
    )


class FeedForwardConfig(BaseModel):
    """Configuration for the Feed-Forward Network layer."""
    d_model: int = Field(..., description="Размерность векторов модели.")
    d_ff: int = Field(..., description="Размерность скрытого слоя.")
    bias: bool = Field(False, description="Использовать ли смещение в линейных слоях.")
    num_layers: int = Field(1, description="Количество подслоев FFN.")
    use_internal_norm: bool = Field(False, description="Использовать ли RMSNorm внутри FFN.")


class DecoderBlockConfig(BaseModel):
    """Configuration for a single DecoderBlock."""
    d_model: int = Field(..., description="Размерность векторов модели.")
    num_heads: int = Field(..., description="Количество голов внимания.")
    d_ff: int = Field(..., description="Размерность скрытого слоя FFN.")
    dropout_rate: float = Field(..., description="Вероятность дропаута.")
    num_kv_heads: int = Field(..., description="Количество голов KV для GQA.")
    num_layers: int = Field(..., description="Общее количество слоев в блоке.")
    num_experts: Optional[int] = Field(None, description="Количество экспертов для MoE.")
    top_k_experts: Optional[int] = Field(
        None, description="Количество выбираемых экспертов на токен."
    )
    use_shared_expert: bool = Field(
        False, description="Использовать ли общий эксперт (shared expert)."
    )
    rotary_emb: Optional[Tuple[Any, Any]] = Field(
        None, description="Кортеж эмбеддингов RoPE."
    )
    long_term_memory: Optional[Any] = Field(None, description="Экземпляр модуля LTM.")
    load_in_4bit: bool = Field(False, description="Использовать ли 4-битную квантование.")

    class Config:  # pylint: disable=too-few-public-methods
        """Pydantic config."""
        arbitrary_types_allowed = True


class LTMArchitectureConfig(BaseModel):
    """Configuration specific to the LTM architecture."""
    d_hidden: int | None = Field(
        None, description="Размерность ассоциативного пространства LTM.")
    num_layers: int | None = Field(
        None, description="Количество слоев в LTM (применимо для MLP-компонентов).")
    num_heads: int = Field(
        1, description="Количество голов в модуле Long-Term Memory.")


class ModelConfig(BaseModel):
    """Configuration for the Transformer model architecture."""
    vocab_size: int | None = Field(
        None, description="Размер словаря.")
    d_model: int = Field(...,
                         description="Размерность векторов модели.")
    num_layers: int = Field(
        ..., description="Количество слоев в энкодере и декодере.")
    num_heads: int = Field(...,
                           description="Количество голов в Multi-Head Attention.")
    num_kv_heads: int = Field(
        ...,
        description="Количество голов для Key/Value в Grouped-Query Attention.")
    d_ff: int = Field(..., description="Размерность Feed-Forward слоев.")
    max_seq_len: int = Field(..., description="Максимальная длина последовательности.")
    dropout_rate: float = Field(..., description="Вероятность дропаута.")
    ltm: LTMArchitectureConfig = Field(
        default_factory=LTMArchitectureConfig,
        description="Конфигурация модуля Long-Term Memory."
    )
    num_experts: int | None = Field(
        None, description="Количество 'экспертов' в слое MoE.")
    top_k_experts: int | None = Field(
        None, description="Количество 'экспертов', выбираемых для каждого токена.")
    use_shared_expert: bool = Field(
        False, description="Использовать ли общий эксперт (shared expert)."
    )
    gradient_checkpointing: bool = Field(
        False, description="Включить чекпоинты градиентов для экономии памяти.")
    anchor_window_size: int = Field(
        4, description="Размер фиксированного окна 'якорей' (Attention Sinks) для контекста."
    )
    rope_ntk_factor: float = Field(
        1.0, description="Фактор масштабирования NTK-aware RoPE для расширения контекста."
    )
    logit_soft_cap: float | None = Field(
        None, description="Порог для мягкого ограничения логитов (logit soft-clamping)."
    )


class VisionConfig(BaseModel):
    """Configuration for the Vision Encoder."""
    image_size: tuple[int, int] = Field(
        (224, 224), description="Размер входного изображения (высота, ширина).")
    patch_size: int = Field(16,
                            description="Размер одного патча изображения.")
    num_channels: int = Field(
        3, description="Количество каналов в изображении (например, 3 для RGB).")


class ComplexityConfig(BaseModel):
    """Configuration for dynamic parameter adjustment."""
    low_complexity_threshold: float = Field(
        0.5, description="Порог для маршрутизации 'низкой' сложности."
    )
    medium_complexity_threshold: float = Field(
        1.5, description="Порог для маршрутизации 'средней' сложности."
    )
    high_complexity_threshold: float = Field(
        3.0, description="Порог для маршрутизации 'высокой' сложности."
    )
    low_complexity_top_k: int = Field(
        1, description="MoE top_k для задач низкой сложности."
    )
    medium_complexity_top_k: int = Field(
        10, description="MoE top_k для задач средней сложности."
    )
    high_complexity_top_k: int = Field(
        50, description="MoE top_k для задач высокой сложности."
    )
