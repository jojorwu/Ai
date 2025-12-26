"""
Tests for the quantization functions.
"""
import unittest

from backend import np
from quantization import dequantize, quantize


class TestQuantization(unittest.TestCase):
    """
    Tests for the quantization functions.
    """

    def test_quantize_dequantize(self):
        """
        Tests that quantizing and then dequantizing a tensor
        returns a result that is close to the original.
        """
        original_weights = (np.random.rand(10, 20) - 0.5).astype(np.float32)
        quantized_weights, scale, _ = quantize(original_weights)
        dequantized_weights = dequantize(quantized_weights, scale)

        self.assertEqual(quantized_weights.dtype, np.int8)
        self.assertEqual(dequantized_weights.dtype, np.float32)
        np.testing.assert_allclose(original_weights, dequantized_weights, atol=1e-2)


if __name__ == '__main__':
    unittest.main()
