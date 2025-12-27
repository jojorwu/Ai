"""
Tests for the PyTorch-based Dropout layer.
"""
import unittest
import torch

from nn_components.dropout import Dropout


class TestDropout(unittest.TestCase):
    """
    Tests for the PyTorch Dropout layer.
    """

    def test_dropout_train_mode(self):
        """Tests that dropout is applied in training mode."""
        probability = 0.5
        dropout = Dropout(probability)
        dropout.train()  # Set the module to training mode

        # Input tensor with large values to make it unlikely that they become zero by chance
        x = torch.ones(100, 100) * 1000

        output = dropout(x)

        # Check that some elements have been zeroed out
        self.assertTrue(torch.any(output == 0))

        # Check that not ALL elements have been zeroed out (sanity check)
        self.assertTrue(torch.any(output != 0))

        # The expected value of the output should be close to the input
        # E[output] = (1-p)*(x / (1-p)) + p*0 = x
        # This is hard to test precisely, so we check if the non-zero elements were scaled.
        # The scaling factor is 1 / (1 - p)
        expected_value = 1 / (1 - probability)
        # Check a non-zero element to see if it was scaled correctly
        non_zero_elements = output[output != 0]
        if len(non_zero_elements) > 0:
            self.assertAlmostEqual(non_zero_elements[0].item(), expected_value * 1000, places=4)

    def test_dropout_eval_mode(self):
        """Tests that dropout is NOT applied in evaluation mode."""
        probability = 0.5
        dropout = Dropout(probability)
        dropout.eval()  # Set the module to evaluation mode

        x = torch.randn(10, 20)
        output = dropout(x)

        # In eval mode, the output should be identical to the input
        self.assertTrue(torch.equal(output, x))


if __name__ == "__main__":
    unittest.main()
