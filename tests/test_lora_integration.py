"""
Integration tests for LoRA functionality.
"""
import unittest

from accelerate import Accelerator

from src.trainer import DataComponents
from tests.test_utils import create_test_config_and_data, run_peft_model_test


class TestLoraIntegration(unittest.TestCase):
    """Integration tests for LoRA functionality."""

    def setUp(self):
        """Set up a test configuration with LoRA enabled."""
        (
            self.config,
            self.tokenizer,
            self.train_data,
            self.val_data,
        ) = create_test_config_and_data(lora_enabled=True)
        self.accelerator = Accelerator()

    def test_lora_model_creation_and_training(self):
        """
        Tests that a model with LoRA can be created, trained for one step,
        and saved correctly.
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
            load_in_4bit=False,
        )

if __name__ == '__main__':
    unittest.main()
