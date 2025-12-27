"""
Tests for the SoftmaxCrossEntropy loss function.
"""
import unittest

import numpy as np

from nn_components.loss import SoftmaxCrossEntropy


class TestSoftmaxCrossEntropy(unittest.TestCase):
    """
    Tests for the SoftmaxCrossEntropy loss function.
    """

    def setUp(self):
        """Set up the test environment."""
        self.loss_fn = SoftmaxCrossEntropy()
        np.random.seed(42)

    def test_dimensions(self):
        """Tests the dimensions of the output and gradient."""
        batch_size, seq_len, vocab_size = 4, 10, 50
        logits = np.random.randn(batch_size, seq_len, vocab_size)
        targets = np.random.randint(0, vocab_size, (batch_size, seq_len))

        loss = self.loss_fn.forward(logits, targets)
        dx = self.loss_fn.backward()

        self.assertIsInstance(loss, float)
        self.assertEqual(dx.shape, logits.shape)

    def test_loss_value(self):
        """Tests the correctness of the calculated loss value."""
        logits_simple = np.array([[[1.0, 2.0, 3.0]]])
        targets_simple = np.array([[2]])
        loss_val = self.loss_fn.forward(logits_simple, targets_simple)
        expected_loss = 0.407
        self.assertTrue(np.isclose(loss_val, expected_loss, atol=1e-3))

    def test_gradient_value(self):
        """Tests that the sum of gradients is close to zero."""
        logits_simple = np.array([[[1.0, 2.0, 3.0]]])
        targets_simple = np.array([[2]])
        self.loss_fn.forward(logits_simple, targets_simple)
        dx_val = self.loss_fn.backward()
        self.assertTrue(np.isclose(np.sum(dx_val), 0, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
