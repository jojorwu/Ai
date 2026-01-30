"""
Pydantic models for training, evolution, and optimizer configuration.
"""
from pydantic import BaseModel, Field


class EvolutionConfig(BaseModel):
    """Configuration for the evolutionary training process."""
    pretrain_epochs: int = Field(
        ..., description="Количество эпох для начального предварительного обучения.")
    evolution_epochs: int = Field(
        ..., description="Количество циклов эволюции (поколений).")
    num_agents: int = Field(...,
                           description="Количество агентов в одной популяции.")
    num_survivors: int = Field(
        ..., description="Количество лучших агентов, чьи LTM будут объединены.")
    batch_size: int = Field(..., description="Размер одного пакета данных.")
    seq_len: int = Field(..., description="Длина последовательности для обучения.")
    gradient_accumulation_steps: int = Field(
        ..., description="Количество шагов для накопления градиентов.")
    validation_split: float = Field(
        ..., description="Доля данных для валидации.")
    data_dir: str = Field(..., description="Директория с данными для обучения.")
    weights_path: str = Field(...,
                              description="Путь для сохранения весов итоговой модели.")
    checkpoint_path: str | None = Field(None,
                                            description="Путь для сохранения чекпоинтов.")
    early_stopping_patience: int = Field(
        3, description="Количество эпох без улучшения для ранней остановки.")
    best_model_path: str = Field("best_model.npz",
                                 description="Путь для сохранения лучшей модели.")
    moe_aux_loss_coeff: float = Field(
        0.01, description="Коэффициент для вспомогательной функции потерь MoE.")
    label_smoothing: float = Field(
        0.0, description="Значение сглаживания меток (0.0 означает выключено).")


class OptimizerConfig(BaseModel):
    """Configuration for the Adam optimizer."""
    learning_rate: float = Field(..., description="Скорость обучения (Learning rate).")
    beta1: float = Field(..., description="Коэффициент Beta1 для Adam.")
    beta2: float = Field(..., description="Коэффициент Beta2 для Adam.")
    epsilon: float = Field(..., description="Малое значение для предотвращения деления на ноль.")
    weight_decay: float = Field(..., description="Коэффициент распада весов (L2 регуляризация).")
    max_norm: float = Field(..., description="Максимальная норма для ограничения градиентов.")


class LTMOptimizerConfig(BaseModel):
    """Configuration for the LTM's Adam optimizer."""
    learning_rate: float = Field(..., description="Скорость обучения для обновлений LTM.")
    beta1: float = Field(0.9, description="Коэффициент Beta1 для оптимизатора Adam.")
    beta2: float = Field(0.999, description="Коэффициент Beta2 для оптимизатора Adam.")
    epsilon: float = Field(1e-8, description="Epsilon для предотвращения деления на ноль.")
    weight_decay: float = Field(0.01, description="Распад весов (L2 регуляризация).")


class LTMConfig(BaseModel):
    """Configuration for the Long-Term Memory (LTM)."""
    surprise_threshold: float = Field(
        ..., description="Порог 'удивления' для обновления LTM.")
    optimizer: LTMOptimizerConfig = Field(
        ..., description="Настройки оптимизатора для модуля LTM."
    )


class SchedulerConfig(BaseModel):
    """Configuration for the learning rate scheduler."""
    warmup_steps: int = Field(...,
                              description="Количество шагов 'разогрева'.")
    training_steps: int = Field(...,
                                description="Общее количество шагов обучения.")
    min_lr: float = Field(..., description="Минимальное значение скорости обучения.")
