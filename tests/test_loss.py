"""
Tests for the SoftmaxCrossEntropy loss function.
"""

import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn_components.loss import SoftmaxCrossEntropy

class TestSoftmaxCrossEntropy(unittest.TestCase):
    """
    Tests for the SoftmaxCrossEntropy loss function.
    """
    def test_softmax_cross_entropy_stable(self):
        """
        Тестирование численно стабильной версии SoftmaxCrossEntropy.
        """
        print("\\nRunning tests for stable SoftmaxCrossEntropy...")

        batch_size, seq_len, vocab_size = 4, 10, 50
        loss_fn = SoftmaxCrossEntropy()

        np.random.seed(42)
        logits = np.random.randn(batch_size, seq_len, vocab_size)
        targets = np.random.randint(0, vocab_size, (batch_size, seq_len))

        loss = loss_fn.forward(logits, targets)
        dx = loss_fn.backward()

        self.assertIsInstance(loss, float, "Test 1 Failed: Loss should be a float.")
        self.assertEqual(dx.shape, logits.shape,
                         f"Test 1 Failed: Grad shape is {dx.shape}, expected {logits.shape}")
        print("Test 1 (Dimensions) PASSED.")

        logits_simple = np.array([[[1.0, 2.0, 3.0]]])
        targets_simple = np.array([[2]])
        loss_val = loss_fn.forward(logits_simple, targets_simple)

        expected_loss = 0.407
        self.assertTrue(np.isclose(loss_val, expected_loss, atol=1e-3),
                        f"Test 2 Failed: Loss is {loss_val}, expected {expected_loss}")
        print("Test 2 (Loss Value) PASSED.")

        dx_val = loss_fn.backward()
        self.assertTrue(np.isclose(np.sum(dx_val), 0, atol=1e-7),
                        f"Test 3 Failed: Sum of grads is {np.sum(dx_val)}, expected 0.")
        print("Test 3 (Gradient Value) PASSED.")

        logits_comp = np.array([[[0.1, 0.1, 0.6, 0.1, 0.1]]])
        targets_comp = np.array([[2]])
        loss_comp_val = loss_fn.forward(logits_comp, targets_comp)
        expected_old_loss = 1.23
        self.assertTrue(np.isclose(loss_comp_val, expected_old_loss, atol=0.01),
                        f"Test 4 Failed: Loss {loss_comp_val} vs expected {expected_old_loss}")
        print("Test 4 (Comparison with old implementation) PASSED.")
        print("All tests for SoftmaxCrossEntropy passed!")

if __name__ == "__main__":
    unittest.main()
