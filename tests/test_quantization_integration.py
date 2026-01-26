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

from src.config.core import TrainConfig
from src.training.trainer import (
    create_trainer,
)
from src.model.factory import load_model_and_tokenizer
from tests.test_utils import MockTokenizer


class TestQuantizationIntegration(unittest.TestCase):
    """
    Tests the full lifecycle of a 4-bit quantized model:
    1. A model is trained for one step and saved.
    2. The saved model is loaded with 4-bit quantization.
    3. A training step is run on the quantized model to ensure compatibility.
    """

    def setUp(self):
        """Set up a temporary directory for model artifacts."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.temp_dir, "test_model")
        os.makedirs(self.model_dir, exist_ok=True)

        # Create dummy data and config
        self.config_path = os.path.join(self.temp_dir, "config.json")
        self.vocab_path = os.path.join(self.temp_dir, "tokenizer_vocab.json")
        self.data_path = os.path.join(self.temp_dir, "data.txt")

        with open("config/config_train.json", "r", encoding="utf-8") as f:
            config = TrainConfig.model_validate_json(f.read())
        config.model.d_model = 16
        config.model.num_heads = 2
        config.model.d_ff = 32
        config.model.num_layers = 1
        config.evolution.data_dir = self.temp_dir
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write(config.model_dump_json(indent=4))

        vocab = {"<pad>": 0, "a": 1, "b": 2}
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f)

        with open(self.data_path, "w", encoding="utf-8") as f:
            f.write("a b " * 50)

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_quantized_model_training_lifecycle(self): # pylint: disable=too-many-locals
        """
        Tests the full training lifecycle with a 4-bit quantized model.
        """
        # 1. Initial training and saving
        config = TrainConfig.from_json(self.config_path)
        accelerator = Accelerator()
        tokenizer = MockTokenizer()
        train_data = torch.randint(
            0, tokenizer.vocab_size, (100,)
        )
        val_data = torch.randint(
            0, tokenizer.vocab_size, (20,)
        )
        trainer = create_trainer(
            config=config,
            tokenizer=tokenizer,
            train_data=train_data,
            val_data=val_data,
            accelerator=accelerator,
        )
        trainer.train_pretrain_epoch()
        # Ensure we save the unwrapped model state dict
        unwrapped_model = accelerator.unwrap_model(trainer.get_model())
        torch.save(
            unwrapped_model.state_dict(),
            os.path.join(self.model_dir, "model.pt")
        )
        shutil.copy(self.config_path, self.model_dir)
        shutil.copy(self.vocab_path, self.model_dir)

        # 2. Load the model with 4-bit quantization
        quantized_model, _ = load_model_and_tokenizer(
            os.path.basename(self.model_dir),
            config,
            load_in_4bit=True,
            quantized=False
        )
        self.assertTrue(
            any(
                "4bit" in str(type(m)).lower()
                for m in quantized_model.modules()
            )
        )

        # 3. Run a training step on the quantized model
        # We need to create a new optimizer for the quantized model's parameters
        quantized_optimizer = torch.optim.AdamW(
            quantized_model.parameters(), lr=config.optimizer.learning_rate
        )

        # We can reuse the scheduler and loss function from the original trainer.
        # We access the internal config of the original trainer to get them.
        # pylint: disable=protected-access
        scheduler = trainer.scheduler
        value_loss_fn = trainer.value_loss_fn

        # Prepare the new model, optimizer, and re-prepare the reused components
        (
            quantized_model,
            quantized_optimizer,
            scheduler,
            value_loss_fn,
        ) = accelerator.prepare(
            quantized_model, quantized_optimizer, scheduler, value_loss_fn
        )

        # Directly create a new trainer with the quantized components
        quantized_trainer = create_trainer(
            config=config,
            tokenizer=tokenizer,
            train_data=train_data,
            val_data=val_data,
            accelerator=accelerator,
            load_in_4bit=True
        )
        loss, _ = quantized_trainer.train_pretrain_epoch()

        # Assert that the training step was successful (loss is a valid number)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))


if __name__ == "__main__":
    unittest.main()
