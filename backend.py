"""
Переключаемый бэкенд для вычислений.
Позволяет выбирать между NumPy (cpu), CuPy (gpu) и будущей поддержкой MPS.
"""
import importlib
import os

# Глобальная переменная для хранения текущего бэкенда.
# Изначально None, устанавливается функцией set_backend.
np = None

def set_backend(device='cpu'):
    """
    Устанавливает и импортирует необходимый бэкенд для вычислений.

    Args:
        device (str): Устройство для вычислений. 'cpu', 'gpu', или 'mps'.
    """
    global np
    if device == 'gpu':
        try:
            # CuPy для вычислений на GPU NVIDIA
            np = importlib.import_module('cupy')
            print("Using CuPy for GPU acceleration")
        except ImportError:
            print("CuPy not found, falling back to NumPy on CPU for 'gpu' device.")
            np = importlib.import_module('numpy')
    elif device == 'mps':
        # В будущем здесь может быть реализация для Apple Silicon (MPS).
        # На данный момент NumPy является наиболее совместимым вариантом.
        print("MPS backend is not yet fully supported, falling back to NumPy on CPU.")
        np = importlib.import_module('numpy')
    elif device == 'cpu':
        # NumPy для вычислений на CPU
        np = importlib.import_module('numpy')
        print("Using NumPy on CPU")
    else:
        raise ValueError(f"Unsupported device: {device}. Choose from 'cpu', 'gpu', 'mps'.")

# Устанавливаем CPU (NumPy) как бэкенд по умолчанию при первом импорте.
# Основные скрипты (train.py, generate.py) должны вызывать set_backend
# с нужным значением из конфига, чтобы переопределить это.
if np is None:
    set_backend('cpu')
