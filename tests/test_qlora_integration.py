"""
Integration tests for QLoRA functionality.
"""
import os
import tempfile
import unittest

import torch
from accelerate import Accelerator
from peft import PeftModel

from src.model.loss import cross_entropy_with_label_smoothing
from src.trainer import create_trainer, DataComponents
from tests.test_utils import create_test_config_and_data


class TestQLoraIntegration(unittest.TestCase):
    """Integration tests for QLoRA functionality."""

    def setUp(self):
        """Set up a test configuration with LoRA enabled."""
        self.config, self.tokenizer, self.train_data, self.val_data = create_test_config_and_data(
            lora_enabled=True
        )
        self.accelerator = Accelerator()

    @unittest.skipIf(not torch.cuda.is_available(), "QLoRA test requires CUDA")
    def test_qlora_model_creation_and_training(self):
        """
        Tests that a 4-bit model with LoRA can be created, trained, and saved.
        """
        data_components = DataComponents(
            tokenizer=self.tokenizer,
            train_data=self.train_data,
            val_data=self.val_data,
        )

        trainer = create_trainer(
            config=self.config,
            data_components=data_components,
            accelerator=self.accelerator,
            load_in_4bit=True,
        )

        model = trainer.get_model()

        # 1. Check if the model is a PeftModel
        unwrapped_model = self.accelerator.unwrap_model(model)
        self.assertIsInstance(unwrapped_model, PeftModel)

        # 2. Check trainable parameters
        total_params = sum(p.numel() for p in unwrapped_model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.assertLess(trainable_params, total_params)
        self.assertGreater(trainable_params, 0)

        # 3. Perform a single training step
        input_ids = torch.tensor([[0, 1, 2]], device=self.accelerator.device)
        labels = torch.tensor([[1, 2, 0]], device=self.accelerator.device)

        # pylint: disable=protected-access
        optimizer = trainer._config.components.optimizer

        logits, _, _ = model(input_ids=input_ids)

        loss = cross_entropy_with_label_smoothing(
            logits,
            labels,
            smoothing=0.0,
            vocab_size=self.tokenizer.vocab_size,
        )

        self.accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        # 4. Check model saving
        with tempfile.TemporaryDirectory() as tmpdir:
            unwrapped_model.save_pretrained(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "adapter_model.safetensors")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "adapter_config.json")))

if __name__ == '__main__':
    unittest.main()
