"""
Tests for the PyTorch-based Transformer DecoderBlock.
"""
import unittest
from unittest.mock import MagicMock

import torch

from src.config import DecoderBlockConfig
from src.nn_components.decoder_block import DecoderBlock, ForwardPassInput


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
            top_k_experts=2,
            long_term_memory=MagicMock(spec=torch.nn.Module)
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
        ltm_state = torch.randn(4, 1, config.d_model, requires_grad=True)
        inputs = ForwardPassInput(x=x, ltm_state=ltm_state)

        # Forward pass
        output, aux_loss = decoder_block(inputs)

        # Simulate a combined loss and backward pass
        fake_loss = output.sum() + aux_loss
        fake_loss.backward()

        # Check gradients for some key parameters
        # MHA gradients
        self.assertIsNotNone(decoder_block.mha.qkv_proj.weight.grad)
        self.assertIsNotNone(decoder_block.mha.wo.weight.grad)

        # Norm gradients
        self.assertIsNotNone(decoder_block.norm['norm1'].gamma.grad)
        self.assertIsNotNone(decoder_block.norm['norm2'].gamma.grad)

        # FiLM gradients
        if decoder_block.ltm:
            self.assertIsNotNone(decoder_block.film1.projection.weight.grad)
            self.assertIsNotNone(decoder_block.film2.projection.weight.grad)

        # MoE/FFN gradients
        if decoder_block.use_moe:
            self.assertIsNotNone(decoder_block.ff_layer.gate.weight.grad)
            self.assertTrue(
                any(
                    p.grad is not None
                    for p in decoder_block.ff_layer.experts[0].parameters()
                )
            )
        else:
            self.assertIsNotNone(decoder_block.ff_layer.w1.weights.grad)

        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
