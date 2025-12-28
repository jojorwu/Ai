"""
Tests for the PyTorch-based Embedding layer.
"""
import unittest
import torch
from nn_components.embedding import Embedding


class TestEmbedding(unittest.TestCase):
    """
    Tests for the PyTorch Embedding layer.
    """

    def test_forward_pass_shape(self):
        """Tests that the forward pass produces the correct output shape."""
        vocab_size = 100
        d_model = 64
        embedding = Embedding(vocab_size, d_model)

        # Input tensor of indices
        x = torch.randint(0, vocab_size, (4, 10))  # Batch, SeqLen

        output = embedding(x)

        self.assertEqual(output.shape, (4, 10, d_model))

    def test_backward_pass_computes_grads(self):
        """
        Tests the backward pass to ensure gradients are computed for the embedding weights.
        """
        vocab_size = 100
        d_model = 64
        embedding = Embedding(vocab_size, d_model)

        # Input tensor of indices
        x = torch.randint(0, vocab_size, (4, 10))

        # Forward pass
        output = embedding(x)

        # Simulate a loss and backward pass
        fake_loss = output.sum()
        fake_loss.backward()

        # Check that gradients exist for the embedding weights
        self.assertIsNotNone(embedding.weights.grad)
        self.assertEqual(embedding.weights.grad.shape, embedding.weights.shape)

        # Gradients should only be non-zero for the indices that were looked up
        # Create a tensor of unique indices from the input
        unique_indices = torch.unique(x)

        # Create a mask for all other indices
        other_indices_mask = torch.ones(vocab_size, dtype=torch.bool)
        other_indices_mask[unique_indices] = False

        # The sum of gradients for indices that were NOT in the input should be zero
        grad_sum_for_other_indices = embedding.weights.grad[other_indices_mask].sum()
        self.assertEqual(grad_sum_for_other_indices, 0)


if __name__ == "__main__":
    unittest.main()
