"""
Pydantic модели для строгой типизации и валидации конфигурации проекта.
"""

import json
from pydantic import BaseModel, Field
from typing import Literal, Optional

class ModelConfig(BaseModel):
    """Конфигурация архитектуры модели Transformer."""
    d_model: int = Field(..., description="Размерность векторов модели.")
    num_layers: int = Field(..., description="Количество слоев в энкодере и декодере.")
    num_heads: int = Field(..., description="Количество голов в Multi-Head Attention.")
    num_kv_heads: int = Field(..., description="Количество голов для Key/Value в Grouped-Query Attention.")
    d_ff: int = Field(..., description="Размерность в Feed-Forward слоях.")
    max_seq_len: int = Field(..., description="Максимальная длина последовательности.")
    dropout_rate: float = Field(..., description="Вероятность отключения нейронов в Dropout слоях.")
    ltm_d_hidden: Optional[int] = Field(None, description="Размерность скрытого слоя в LTM.")
    ltm_num_layers: Optional[int] = Field(None, description="Количество слоев в LTM.")
    num_experts: Optional[int] = Field(None, description="Количество 'экспертов' в слое MoE.")
    top_k_experts: Optional[int] = Field(None, description="Количество выбираемых 'экспертов' для каждого токена.")

class VisionConfig(BaseModel):
    """Конфигурация для Vision Encoder."""
    image_size: tuple[int, int] = Field((224, 224), description="Размер входного изображения (высота, ширина).")
    patch_size: int = Field(16, description="Размер одного 'патча' изображения.")
    num_channels: int = Field(3, description="Количество каналов в изображении (например, 3 для RGB).")

class EvolutionConfig(BaseModel):
    """Конфигурация эволюционного процесса обучения."""
    pretrain_epochs: int = Field(..., description="Количество эпох для начального предобучения базовой модели.")
    evolution_epochs: int = Field(..., description="Количество циклов (поколений) эволюции агентов.")
    num_agents: int = Field(..., description="Количество агентов в одной популяции.")
    num_survivors: int = Field(..., description="Количество лучших агентов, чьи LTM будут объединены.")

    batch_size: int = Field(..., description="Размер одного батча.")
    seq_len: int = Field(..., description="Длина последовательности для обучения.")
    gradient_accumulation_steps: int = Field(..., description="Количество шагов для накопления градиентов.")
    validation_split: float = Field(..., description="Доля данных для валидации.")

    data_dir: str = Field(..., description="Директория с обучающими данными.")
    weights_path: str = Field(..., description="Путь для сохранения финальных весов модели.")
    checkpoint_path: Optional[str] = Field(None, description="Путь для сохранения чекпоинтов.")
    early_stopping_patience: int = Field(3, description="Количество эпох без улучшения для ранней остановки.")
    best_model_path: str = Field("best_model.npz", description="Путь для сохранения лучшей модели.")
    moe_aux_loss_coeff: float = Field(0.01, description="Коэффициент для вспомогательной 'балансировочной' потери MoE.")

class OptimizerConfig(BaseModel):
    """Конфигурация оптимизатора Adam."""
    learning_rate: float = Field(..., description="Скорость обучения.")
    beta1: float = Field(..., description="Коэффициент beta1 для Adam.")
    beta2: float = Field(..., description="Коэффициент beta2 для Adam.")
    epsilon: float = Field(..., description="Малое значение для предотвращения деления на ноль.")
    weight_decay: float = Field(..., description="Коэффициент затухания весов (L2 регуляризация).")
    max_norm: float = Field(..., description="Максимальная норма для обрезки градиентов.")

class LTMOptimizerConfig(BaseModel):
    """Конфигурация оптимизатора Adam для LTM."""
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float

class LTMConfig(BaseModel):
    """Конфигурация долгосрочной памяти (LTM)."""
    surprise_threshold: float = Field(..., description="Порог 'удивления' для обновления LTM.")
    optimizer: LTMOptimizerConfig

class SchedulerConfig(BaseModel):
    """Конфигурация планировщика скорости обучения."""
    warmup_steps: int = Field(..., description="Количество шагов для 'прогрева'.")
    min_lr: float = Field(..., description="Минимальное значение скорости обучения.")

class GenerationConfig(BaseModel):
    """Конфигурация процесса генерации текста."""
    start_text: str = Field(..., description="Начальный текст для генерации.")
    max_len: int = Field(..., description="Максимальная длина генерируемого текста.")
    temperature: float = Field(..., description="Температура для сэмплирования.")
    top_k: int = Field(..., description="Top-k для сэмплирования.")
    top_p: float = Field(..., description="Top-p (nucleus) для сэмплирования.")
    speculative_steps: int = Field(..., description="Количество спекулятивных шагов.")
    value_threshold: float = Field(..., description="Порог значения для принятия спекулятивной генерации.")
    max_thought_len: int = Field(..., description="Максимальная длина 'мыслей'.")
    max_retries: int = Field(..., description="Максимальное количество попыток при неудачной спекуляции.")
    max_turns: int = Field(10, description="Максимальное количество итераций (вызовов инструментов) в цикле агента.")

class HardwareConfig(BaseModel):
    """Конфигурация оборудования."""
    device: Literal["cpu", "gpu", "mps"] = Field("cpu", description="Устройство для вычислений (cpu, gpu, mps).")

class Config(BaseModel):
    """Основная конфигурационная модель."""
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
        """Загружает конфигурацию из JSON файла и валидирует ее."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
