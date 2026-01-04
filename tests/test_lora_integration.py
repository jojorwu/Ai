import unittest
from unittest.mock import MagicMock
import torch
from accelerate import Accelerator
from src.config import TrainConfig, LoraConfig
from src.trainer import create_trainer, DataComponents
from src.data.tokenizer import Tokenizer

class TestLoraIntegration(unittest.TestCase):
    def setUp(self):
        # Mock Tokenizer
        self.tokenizer = Tokenizer('data')
        self.tokenizer.vocab = {'a': 0, 'b': 1, 'c': 2, '<pad>': 3}
        self.tokenizer.reverse_vocab = {0: 'a', 1: 'b', 2: 'c', 3: '<pad>'}

        # Mock Data
        self.train_data = [0, 1, 2, 0, 1, 2]
        self.val_data = [2, 1, 0]

        # Mock Config
        self.config = TrainConfig.from_json('config_train.json')
        self.config.lora = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["qkv_proj"],
            lora_dropout=0.1,
            bias="none"
        )
        # Use a smaller model for testing
        self.config.model.d_model = 16
        self.config.model.num_heads = 2
        self.config.model.num_kv_heads = 2
        self.config.model.d_ff = 32
        self.config.model.num_layers = 1

        self.accelerator = Accelerator()

    def test_lora_model_creation_and_training(self):
        data_components = DataComponents(
            tokenizer=self.tokenizer,
            train_data=self.train_data,
            val_data=self.val_data
        )

        trainer = create_trainer(
            config=self.config,
            data_components=data_components,
            accelerator=self.accelerator,
            load_in_4bit=False
        )

        model = trainer.get_model()

        # 1. Check if the model is a PeftModel
        from peft import PeftModel

        unwrapped_model = self.accelerator.unwrap_model(model)
        self.assertIsInstance(unwrapped_model, PeftModel)

        # 2. Check trainable parameters
        unwrapped_model.print_trainable_parameters()
        total_params = sum(p.numel() for p in unwrapped_model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.assertLess(trainable_params, total_params)
        self.assertGreater(trainable_params, 0)

        # 3. Perform a single training step
        # Create some dummy data
        input_ids = torch.tensor([[0, 1, 2]], device=self.accelerator.device)
        labels = torch.tensor([[1, 2, 0]], device=self.accelerator.device)

        # Get the optimizer from the trainer
        optimizer = trainer._config.components.optimizer

        from src.model.loss import cross_entropy_with_label_smoothing
        # Forward pass
        logits, _, _ = model(input_ids=input_ids)

        # Calculate loss
        loss = cross_entropy_with_label_smoothing(
            logits,
            labels,
            smoothing=0.0,
            vocab_size=self.tokenizer.vocab_size,
        )

        # Backward pass and optimizer step
        self.accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        # 4. Check model saving
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            unwrapped_model.save_pretrained(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "adapter_model.safetensors")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "adapter_config.json")))

if __name__ == '__main__':
    unittest.main()
