"""
Switchable backend for computations.
Allows choosing between NumPy (cpu), CuPy (gpu), and future MPS support.
"""
import importlib
import logging

# Global variable to hold the current backend.
# Initially None, set by the set_backend function.
np = None


def set_backend(device='cpu'):
    """
    Sets and imports the required backend for computations.
    Args:
        device (str): The device for computations. 'cpu', 'gpu', or 'mps'.
    """
    global np
    if device == 'gpu':
        try:
            np = importlib.import_module('cupy')
            logging.info("Using CuPy for GPU acceleration")
        except ImportError:
            logging.warning("CuPy not found, falling back to NumPy on CPU for 'gpu' device.")
            np = importlib.import_module('numpy')
            np.set_printoptions(precision=4, suppress=True)
            np.float = np.float32
    elif device == 'mps':
        logging.warning("MPS backend is not yet fully supported, falling back to NumPy on CPU.")
        np = importlib.import_module('numpy')
        np.set_printoptions(precision=4, suppress=True)
        np.float = np.float32
    elif device == 'cpu':
        np = importlib.import_module('numpy')
        np.set_printoptions(precision=4, suppress=True)
        np.float = np.float32
        logging.info("Using NumPy on CPU with float32 precision")
    else:
        raise ValueError(f"Unsupported device: {device}. Choose from 'cpu', 'gpu', 'mps'.")


# Set CPU (NumPy) as the default backend on first import.
# Main scripts (train.py, generate.py) should call set_backend
# with the value from the config to override this.
if np is None:
    set_backend('cpu')
