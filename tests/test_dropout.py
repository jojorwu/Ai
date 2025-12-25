"""
Тесты для `Dropout` слоя.
"""
import unittest

from backend import np
from nn_components.dropout import Dropout


class TestDropout(unittest.TestCase):
    """Тестирование слоя Dropout."""

    def test_dropout(self):
        """Тестирование слоя Dropout."""
        print("\\nRunning tests for Dropout...")

        rate = 0.5
        dropout = Dropout(rate)
        x = np.random.randn(10, 20)

        # 1. Тест в режиме обучения
        dropout.train()
        output_train = dropout.forward(x)
        self.assertTrue(np.any(output_train == 0), "Test 1 FAILED: No zeros in train mode output.")
        self.assertAlmostEqual(np.mean(output_train), np.mean(x), delta=0.2,
                             msg="Test 1 FAILED: Mean of output is too different in train mode.")
        print("Test 1 (Train Mode) PASSED.")

        # 2. Тест в режиме оценки
        dropout.eval()
        output_eval = dropout.forward(x)
        self.assertTrue(np.array_equal(output_eval, x),
                        "Test 2 FAILED: Output is not identical to input in eval mode.")
        print("Test 2 (Eval Mode) PASSED.")

if __name__ == "__main__":
    unittest.main()
