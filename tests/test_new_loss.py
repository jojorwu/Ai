import numpy as np
import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_components.loss import MSELoss, MarginRankingLoss

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

    def test_margin_ranking_loss_forward(self):
        """Test the forward pass of the MarginRankingLoss function."""
        margin_loss = MarginRankingLoss(margin=1.0)
        y_good = np.array([2.0])
        y_bad = np.array([0.5])
        # loss = max(0, 1.0 - (2.0 - 0.5)) = max(0, -0.5) = 0
        self.assertAlmostEqual(margin_loss.forward(y_good, y_bad), 0.0)

        y_good = np.array([0.5])
        y_bad = np.array([2.0])
        # loss = max(0, 1.0 - (0.5 - 2.0)) = max(0, 2.5) = 2.5
        self.assertAlmostEqual(margin_loss.forward(y_good, y_bad), 2.5)

    def test_margin_ranking_loss_backward(self):
        """Test the backward pass of the MarginRankingLoss function."""
        margin_loss = MarginRankingLoss(margin=1.0)
        y_good = np.array([2.0])
        y_bad = np.array([0.5])
        margin_loss.forward(y_good, y_bad)
        d_good, d_bad = margin_loss.backward()
        self.assertAlmostEqual(d_good, 0.0)
        self.assertAlmostEqual(d_bad, 0.0)

        y_good = np.array([0.5])
        y_bad = np.array([2.0])
        margin_loss.forward(y_good, y_bad)
        d_good, d_bad = margin_loss.backward()
        self.assertAlmostEqual(d_good, -1.0)
        self.assertAlmostEqual(d_bad, 1.0)

if __name__ == "__main__":
    unittest.main()
