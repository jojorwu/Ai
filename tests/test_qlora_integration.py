"""
Integration tests for QLoRA functionality.
"""
import unittest

import torch
from accelerate import Accelerator

from src.trainer import DataComponents
from tests.test_utils import create_test_config_and_data, run_peft_model_test


class TestQLoraIntegration(unittest.TestCase):
    """Integration tests for QLoRA functionality."""

    def setUp(self):
        """Set up a test configuration with LoRA enabled."""
        (
            self.config,
            self.tokenizer,
            self.train_data,
            self.val_data,
        ) = create_test_config_and_data(lora_enabled=True)
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
        run_peft_model_test(
            self,
            self.config,
            data_components,
            self.accelerator,
            load_in_4bit=True,
        )

if __name__ == '__main__':
    unittest.main()
