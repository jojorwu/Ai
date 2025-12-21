import numpy as np
import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn_components.dropout import Dropout

class TestDropout(unittest.TestCase):
    def test_dropout(self):
        """Тестирование слоя Dropout."""
        print("Running tests for Dropout...")

        dropout_rate = 0.5
        layer = Dropout(dropout_rate)

        np.random.seed(42)
        x = np.random.randn(10, 10)

        # --- Тест 1: Режим обучения (train mode) ---
        layer.is_training = True
        output_train = layer.forward(x)

        # Проверяем, что некоторые нейроны обнулились
        self.assertTrue(np.any(output_train == 0), "Test 1 FAILED: No neurons were dropped out in train mode.")
        # Проверяем, что выход не идентичен входу
        self.assertFalse(np.array_equal(output_train, x), "Test 1 FAILED: Output is identical to input in train mode.")
        print("Test 1 (Train Mode) PASSED.")

        # --- Тест 2: Режим генерации (eval mode) ---
        layer.is_training = False
        output_eval = layer.forward(x)

        # Проверяем, что выход идентичен входу
        self.assertTrue(np.array_equal(output_eval, x), "Test 2 FAILED: Output is not identical to input in eval mode.")
        print("Test 2 (Eval Mode) PASSED.")

        # --- Тест 3: Обратный проход ---
        dout = np.ones_like(x)
        dx = layer.backward(dout)

        # Градиент должен быть равен маске (так как dout - единицы)
        expected_dx = layer.mask
        self.assertTrue(np.array_equal(dx, expected_dx), "Test 3 FAILED: Backward pass is incorrect.")
        print("Test 3 (Backward Pass) PASSED.")

        print("All tests passed!")

if __name__ == "__main__":
    unittest.main()
