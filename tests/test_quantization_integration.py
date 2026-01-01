"""
Integration test for 4-bit quantization.
"""
import os
import unittest
import torch
from accelerate import Accelerator, dispatch_model, init_empty_weights
from bitsandbytes.nn import Linear4bit
from bitsandbytes.optim import Adam8bit
from torch import nn

from src.config import Config, TransformerConfig
from src.model.model import Transformer
from src.utils.core import load_model_and_tokenizer

class TestQuantizationIntegration(unittest.TestCase):
    """
    Tests that a 4-bit quantized model can be loaded, perform a forward pass,
    and complete a training step.
    """

    def setUp(self):
        """Set up a dummy model and config for the tests."""
        self.model_name = "test_model_4bit"
        self.model_dir = os.path.join("models", self.model_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.vocab_size = 100

        self.config_path = os.path.join(self.model_dir, "config.json")
        with open(self.config_path, "w") as f:
            f.write(f"""
            {{
                "vocab_size": {self.vocab_size},
                "model": {{
                    "d_model": 64, "num_layers": 2, "num_heads": 2,
                    "num_kv_heads": 2, "d_ff": 128, "max_seq_len": 128,
                    "dropout_rate": 0.1, "ltm": {{}}, "num_experts": null, "top_k_experts": null
                }},
                "vision": {{}}, "evolution": {{}},
                "optimizer": {{"learning_rate": 0.001}},
                "ltm": {{"surprise_threshold": 0.5, "optimizer": {{}}}},
                "scheduler": {{}}, "generation": {{}}, "hardware": {{}}
            }}
            """)

        vocab_path = os.path.join(self.model_dir, "tokenizer_vocab.json")
        with open(vocab_path, "w") as f:
            f.write('{"<PAD>": 0, "a": 1, "b": 2}')

        self.weights_path = os.path.join(self.model_dir, "model.pt")
        config = Config.from_json(self.config_path)
        transformer_config = TransformerConfig(
            vocab_size=self.vocab_size, model=config.model, vision=config.vision, ltm=config.ltm
        )
        model = Transformer(transformer_config)
        torch.save(model.state_dict(), self.weights_path)

    def tearDown(self):
        """Clean up the dummy model files."""
        for path in [self.config_path, os.path.join(self.model_dir, "tokenizer_vocab.json"), self.weights_path]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.model_dir):
            os.rmdir(self.model_dir)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available, skipping 4-bit test.")
    def test_load_and_forward_in_4bit(self):
        """
        Tests that the model can be loaded in 4-bit mode and perform a forward pass.
        """
        config = Config.from_json(self.config_path)
        model, _ = load_model_and_tokenizer(
            self.model_name, config, load_in_4bit=True, quantized=False
        )

        self.assertTrue(
            any(isinstance(m, Linear4bit) for m in model.modules()),
            "Model should contain Linear4bit layers."
        )

        input_tensor = torch.tensor([[1, 2]], device=model.device)
        with torch.no_grad():
            logits, _, _ = model.forward(input_tensor)

        self.assertIsNotNone(logits)
        self.assertEqual(logits.shape[-1], self.vocab_size)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available, skipping 4-bit test.")
    def test_4bit_model_training_step(self):
        """
        Verifies that a 4-bit model's weights are updated after one training step.
        """
        config = Config.from_json(self.config_path)
        model, _ = load_model_and_tokenizer(
            self.model_name, config, load_in_4bit=True, quantized=False
        )

        optimizer = Adam8bit(model.parameters(), lr=config.optimizer.learning_rate)
        policy_loss_fn = nn.CrossEntropyLoss()

        initial_weights = model.layers.decoder[0].attention_sublayer.mha.wo.weight.clone().detach()

        dummy_input = torch.randint(0, self.vocab_size, (2, 4), device=model.device)
        dummy_target = torch.randint(0, self.vocab_size, (2, 4), device=model.device)

        model.train()
        optimizer.zero_grad()
        logits, _, _ = model(dummy_input)
        loss = policy_loss_fn(logits.view(-1, self.vocab_size), dummy_target.view(-1))
        loss.backward()
        optimizer.step()

        updated_weights = model.layers.decoder[0].attention_sublayer.mha.wo.weight.clone().detach()
        self.assertFalse(
            torch.equal(initial_weights, updated_weights),
            "Model weights were NOT updated after a training step."
        )

if __name__ == "__main__":
    unittest.main()
