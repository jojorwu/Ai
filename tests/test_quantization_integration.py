"""
Integration test to verify the end-to-end training process of a 4-bit quantized model.
"""
import logging
import unittest

import torch
from accelerate import Accelerator, dispatch_model, init_empty_weights
from bitsandbytes.optim import Adam8bit
from torch import nn

from config import Config, TransformerConfig
from model import Transformer


class TestQuantizationIntegration(unittest.TestCase):
    """
    Tests that a 4-bit quantized model can complete a training step and update its weights.
    """

    def setUp(self):
        """Set up the test environment."""
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s'
        )
        torch.cuda.empty_cache()
        self.accelerator = Accelerator()
        self.config = Config.from_json('config.json')
        self.vocab_size = 100

    def _init_model(self):
        """Initializes the 4-bit quantized model."""
        transformer_config = TransformerConfig(
            vocab_size=self.vocab_size, model=self.config.model,
            vision=self.config.vision, ltm=self.config.ltm
        )
        with init_empty_weights():
            model = Transformer(transformer_config, load_in_4bit=True)
        device_map = {"": 0}
        logging.info("Using manual device map: %s", device_map)
        return dispatch_model(model, device_map=device_map)

    def _prepare_training(self, model):
        """Prepares the optimizer and loss functions."""
        optimizer = Adam8bit(
            model.parameters(), lr=self.config.optimizer.learning_rate
        )
        loss_fns = (nn.CrossEntropyLoss(), nn.MSELoss())
        return self.accelerator.prepare(optimizer, *loss_fns)

    def _get_dummy_data(self):
        """Generates dummy data for the training step."""
        batch_size = 4
        seq_len = self.config.model.max_seq_len
        dummy_input = torch.randint(
            0, self.vocab_size, (batch_size, seq_len), device='cpu'
        )
        dummy_policy_target = torch.randint(
            0, self.vocab_size, (batch_size, seq_len), device='cpu'
        )
        dummy_value_target = torch.randn(batch_size, 1, device='cpu')
        return self.accelerator.prepare(
            dummy_input, dummy_policy_target, dummy_value_target
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available, skipping 4-bit test")
    def test_4bit_model_training_step(self):
        """
        Verifies that a 4-bit model's weights are updated after one training step.
        """
        model = self._init_model()
        optimizer, policy_loss_fn, value_loss_fn = self._prepare_training(model)
        dummy_input, dummy_policy_target, dummy_value_target = self._get_dummy_data()

        initial_weights = model.decoder_blocks[0].mha.wo.weight.clone().detach()

        model.train()
        optimizer.zero_grad()
        logits, value, aux_loss = model(dummy_input)
        loss_policy = policy_loss_fn(
            logits.view(-1, self.vocab_size), dummy_policy_target.view(-1)
        )
        loss_value = value_loss_fn(value, dummy_value_target)
        total_loss = loss_policy + loss_value + (aux_loss or 0)
        self.accelerator.backward(total_loss)
        optimizer.step()

        updated_weights = model.decoder_blocks[0].mha.wo.weight.clone().detach()
        weights_updated = not torch.equal(initial_weights, updated_weights)

        self.assertTrue(
            weights_updated, "Model weights were NOT updated after a training step."
        )
        logging.info(
            "Verification PASSED: 4-bit model weights were updated successfully."
        )

if __name__ == "__main__":
    unittest.main()
