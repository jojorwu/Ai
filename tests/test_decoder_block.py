"""
Tests for the PyTorch-based Transformer DecoderBlock.
"""
import sys
import os
import unittest
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import DecoderBlockConfig
from nn_components.decoder_block import DecoderBlock, ForwardPassInput


class TestDecoderBlock(unittest.TestCase):
    """
    Tests for the PyTorch DecoderBlock layer.
    """

    def _get_config(self):
        return DecoderBlockConfig(
            d_model=64,
            num_heads=4,
            num_kv_heads=2,
            d_ff=128,
            dropout_rate=0.1,
            num_layers=2,
            num_experts=4,
            top_k_experts=2
        )

    def test_forward_pass_shape(self):
        """Tests that the forward pass preserves the tensor shape."""
        config = self._get_config()
        decoder_block = DecoderBlock(config)

        x = torch.randn(4, 10, config.d_model) # Batch, SeqLen, Dim
        inputs = ForwardPassInput(x=x, ltm_state=torch.zeros_like(x))

        output, aux_loss = decoder_block(inputs)

        self.assertEqual(x.shape, output.shape)
        self.assertIsNotNone(aux_loss)

    def test_backward_pass_computes_grads(self):
        """
        Tests that gradients are computed for all parameters in the DecoderBlock.
        """
        config = self._get_config()
        decoder_block = DecoderBlock(config)

        x = torch.randn(4, 10, config.d_model, requires_grad=True)
        inputs = ForwardPassInput(x=x, ltm_state=torch.zeros_like(x))

        # Forward pass
        output, aux_loss = decoder_block(inputs)

        # Simulate a combined loss and backward pass
        fake_loss = output.sum() + aux_loss
        fake_loss.backward()

        # Check gradients for some key parameters
        # MHA gradients
        self.assertIsNotNone(decoder_block.mha.qkv_proj.weights.grad)
        self.assertIsNotNone(decoder_block.mha.wo.weights.grad)

        # Norm gradients
        self.assertIsNotNone(decoder_block.norm1.gamma.grad)
        self.assertIsNotNone(decoder_block.norm2.gamma.grad)

        # MoE/FFN gradients
        if decoder_block.use_moe:
            self.assertIsNotNone(decoder_block.moe_layer.gate.weights.grad)
            self.assertTrue(any(p.grad is not None for p in decoder_block.moe_layer.experts[0].parameters()))
        else:
            self.assertIsNotNone(decoder_block.ffn.w1.weights.grad)

        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
