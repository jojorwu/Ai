"""
This module implements quantization functions for model weights.
"""
from backend import np


def quantize(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantizes float32 weights to int8.
    """
    scale = np.max(np.abs(weights), axis=-1, keepdims=True) / 127.0
    quantized_weights = (weights / scale).astype(np.int8)
    return quantized_weights, scale.astype(np.float32), np.zeros_like(scale, dtype=np.float32)


def dequantize(quantized_weights: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Dequantizes int8 weights back to float32.
    """
    return quantized_weights.astype(np.float32) * scale
