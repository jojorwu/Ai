"""
Переключаемый бэкенд для вычислений.
По умолчанию используется NumPy. Если доступен CuPy, используется он.
"""
# try:
#     import cupy as np
#     print("Using CuPy for GPU acceleration")
# except ImportError:
import numpy as np
print("CuPy not found, falling back to NumPy on CPU")
