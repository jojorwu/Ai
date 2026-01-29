"""
Pydantic models for hardware strategy configuration.
"""
from typing import Literal
from pydantic import BaseModel, Field


class HardwareConfig(BaseModel):
    """Hardware configuration."""
    device: Literal["cpu", "gpu", "mps"] = Field(
        "cpu", description="Устройство для вычислений (cpu, gpu, mps).")
    strategy: Literal["unified", "discrete", "hybrid"] = Field(
        "discrete", description="Стратегия использования памяти (unified, discrete, hybrid).")
    torch_compile: bool = Field(
        False, description="Включить torch.compile для ускорения модели.")
    num_threads: int = Field(
        0, description="Количество потоков для внутриоперационного параллелизма (0 — по умолчанию).")
    num_interop_threads: int = Field(
        0, description="Количество потоков для меж-операционного параллелизма (0 — по умолчанию).")
    enable_mkldnn: bool = Field(
        True, description="Включить оптимизации oneDNN (MKLDNN) для CPU.")
    flush_denormals: bool = Field(
        False, description="Включить обнуление денормализованных чисел на CPU.")
    num_workers: int = Field(
        4, description="Количество рабочих процессов для загрузки данных.")
    pin_memory: bool = Field(
        True, description="Включить закрепленную память для ускорения передачи данных CPU->GPU.")
    dynamic_quantization: bool = Field(
        False, description="Включить динамическое квантование (Linear слои в qint8) для CPU.")
