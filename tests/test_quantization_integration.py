"""
Integration test for 4-bit quantization.
"""
import json
import os
import shutil
import tempfile
import unittest

import torch
from accelerate import Accelerator

from src.config import TrainConfig
from src.data.tokenizer import Tokenizer
from src.trainer import DataComponents, create_trainer


class TestQuantizationIntegration(unittest.TestCase):
    """
    Tests that a 4-bit quantized model can be successfully created and trained
    for one epoch.
    """

    def setUp(self):
        """Set up a temporary directory for model artifacts."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_name = "test_quantized_model"
        self.model_dir = os.path.join("models", self.model_name)
        os.makedirs(self.model_dir, exist_ok=True)

        # Create a dummy config and tokenizer vocab
        with open("config_train.json", "r", encoding="utf-8") as f:
            self.config = TrainConfig.model_validate_json(f.read())
        self.config.model.d_model = 16
        self.config.model.num_heads = 2
        self.config.model.d_ff = 32
        self.config.model.num_layers = 1
        with open(
            os.path.join(self.model_dir, "config.json"), "w", encoding="utf-8"
        ) as f:
            f.write(self.config.model_dump_json(indent=4))

        self.tokenizer = Tokenizer(self.config.evolution.data_dir)
        self.tokenizer.save_vocab(self.model_dir)

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir)
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_quantized_model_training_lifecycle(self):
        """
        Tests that a 4-bit quantized model can be created and trained.
        """
        # 1. Setup
        accelerator = Accelerator()
        train_data = list(range(100))
        val_data = list(range(20))
        data_components = DataComponents(
            tokenizer=self.tokenizer, train_data=train_data, val_data=val_data
        )

        # Create a dummy model weights file to load from
        # In a real scenario, this would be from a previous training run
        dummy_model = create_trainer(
            self.config, data_components, accelerator, load_in_4bit=False
        ).get_model()
        unwrapped_model = accelerator.unwrap_model(dummy_model)
        torch.save(
            unwrapped_model.state_dict(), os.path.join(self.model_dir, "model.pt")
        )

        # 2. Create a new trainer with 4-bit loading enabled
        # This will load the dummy weights into a quantized model
        quantized_trainer = create_trainer(
            self.config, data_components, accelerator, load_in_4bit=True
        )

        # 3. Run a training epoch on the quantized model
        loss, _ = quantized_trainer.train_pretrain_epoch()

        # 4. Assert that the training step was successful
        self.assertFalse(torch.isnan(torch.tensor(loss)))
        self.assertFalse(torch.isinf(torch.tensor(loss)))
        self.assertGreater(loss, 0)


if __name__ == "__main__":
    unittest.main()
