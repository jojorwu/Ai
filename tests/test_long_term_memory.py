"""
Tests for the PyTorch-based LongTermMemory module.
"""
import sys
import os
import unittest
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn_components.long_term_memory import LongTermMemory


class TestLongTermMemory(unittest.TestCase):
    """
    Tests for the PyTorch LongTermMemory module.
    """

    def test_forward_pass_shape(self):
        """Tests that the forward pass produces the correct output shape."""
        d_model, d_hidden, num_layers = 64, 128, 2
        ltm = LongTermMemory(d_model, d_hidden, num_layers)

        x = torch.randn(4, 1, d_model)  # Batch, SeqLen (1 for summary), Dim
        output = ltm(x)

        self.assertEqual(output.shape, x.shape)

    def test_backward_pass_computes_grads(self):
        """
        Tests that gradients are computed for all parameters in the LTM.
        """
        d_model, d_hidden, num_layers = 64, 128, 2
        ltm = LongTermMemory(d_model, d_hidden, num_layers)

        x = torch.randn(4, 1, d_model, requires_grad=True)

        # Forward pass
        output = ltm(x)

        # Simulate a loss and backward pass
        fake_loss = output.sum()
        fake_loss.backward()

        # Check that gradients exist for all parameters in the network
        for param in ltm.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.all(param.grad == 0))

        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
