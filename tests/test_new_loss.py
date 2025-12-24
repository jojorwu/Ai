"""
Tests for the MarginRankingLoss function.
"""
import unittest

import numpy as np

from nn_components.loss import MarginRankingLoss


class TestNewLoss(unittest.TestCase):
    """
    Tests for the MarginRankingLoss function.
    """

    def test_margin_ranking_loss_forward(self):
        """Test the forward pass of the MarginRankingLoss function."""
        margin_loss = MarginRankingLoss(margin=1.0)
        y_good = np.array([2.0])
        y_bad = np.array([0.5])
        self.assertAlmostEqual(margin_loss.forward(y_good, y_bad), 0.0)

        y_good = np.array([0.5])
        y_bad = np.array([2.0])
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
