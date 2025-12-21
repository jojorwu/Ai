import numpy as np
import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_components.loss import MSELoss

class TestNewLoss(unittest.TestCase):
    def test_mse_loss_forward(self):
        """Test the forward pass of the MSELoss function."""
        mse = MSELoss()
        y_pred = np.array([1, 2, 3])
        y_true = np.array([1.5, 2.5, 3.5])
        expected = np.mean((y_pred - y_true)**2)
        self.assertAlmostEqual(mse.forward(y_pred, y_true), expected)

    def test_mse_loss_backward(self):
        """Test the backward pass of the MSELoss function."""
        mse = MSELoss()
        y_pred = np.array([1, 2, 3])
        y_true = np.array([1.5, 2.5, 3.5])
        mse.forward(y_pred, y_true)
        expected = 2 * (y_pred - y_true) / y_true.size
        self.assertTrue(np.allclose(mse.backward(), expected))

if __name__ == "__main__":
    unittest.main()
