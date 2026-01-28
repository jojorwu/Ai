"""
Unit tests for the ValueHead.
"""
import unittest
import torch
from src.model.layers.value_head import ValueHead


class TestValueHead(unittest.TestCase):
    """Tests for the ValueHead class."""

    def test_forward_shape(self):
        """Tests the forward pass shape."""
        d_model = 64
        value_head = ValueHead(d_model)

        h = torch.randn(4, d_model)
        value = value_head(h)

        # value shape: (batch, 1)
        self.assertEqual(value.shape, (4, 1))

    def test_backward(self):
        """Tests that gradients flow through the ValueHead."""
        d_model = 64
        value_head = ValueHead(d_model)

        h = torch.randn(2, d_model, requires_grad=True)
        value = value_head(h)
        loss = value.sum()
        loss.backward()

        self.assertIsNotNone(h.grad)
        for param in value_head.parameters():
            self.assertIsNotNone(param.grad)


if __name__ == "__main__":
    unittest.main()
