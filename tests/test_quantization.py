"""
Tests for the model quantization process.
"""
import json
import os
import shutil
import sys
import unittest
from unittest.mock import patch

import numpy as np

# Add project root to path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config, TransformerConfig
from model import Transformer
from nn_components.decoder_block import ForwardPassInput
from quantize_model import main as quantize_main
from tokenizer import Tokenizer


class TestQuantization(unittest.TestCase):
    """Test case for model quantization."""

    def setUp(self):
        """Set up a temporary directory and dummy files for testing."""
        self.temp_dir = "temp_test_quantization"
        os.makedirs(self.temp_dir, exist_ok=True)

        self.config_dict = {
            "model": {
                "d_model": 32, "num_layers": 2, "num_heads": 4, "d_ff": 64,
                "max_seq_len": 128, "dropout_rate": 0.0, "num_kv_heads": 2,
                "num_experts": 2, "top_k_experts": 1, "ltm_d_hidden": 16, "ltm_num_layers": 1
            },
            "vision": {"patch_size": 16, "num_channels": 3},
            "evolution": {
                "pretrain_epochs": 1, "evolution_epochs": 1, "num_agents": 2,
                "num_survivors": 1, "batch_size": 2, "seq_len": 16,
                "gradient_accumulation_steps": 1, "validation_split": 0.1,
                "data_dir": "data/", "weights_path": "models/", "checkpoint_path": "checkpoints/"
            },
            "optimizer": {
                "learning_rate": 1e-4, "beta1": 0.9, "beta2": 0.99, "epsilon": 1e-8,
                "weight_decay": 0.01, "max_norm": 1.0
            },
            "ltm": {
                "surprise_threshold": 0.5,
                "optimizer": {"learning_rate": 1e-4, "beta1": 0.9, "beta2": 0.99, "epsilon": 1e-8, "weight_decay": 0.0}
            },
            "scheduler": {"warmup_steps": 10, "min_lr": 1e-5},
            "generation": {
                "start_text": "a", "max_len": 10, "temperature": 0.7, "top_k": 5, "top_p": 0.9,
                "speculative_steps": 2, "value_threshold": -1.0, "max_retries": 3, "max_thought_len": 5
            },
            "hardware": {"device": "cpu"}
        }
        self.config = Config(**self.config_dict)

        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.vocab_path = os.path.join(self.data_dir, "vocab.txt")
        with open(self.vocab_path, 'w', encoding='utf-8') as f:
            f.write("abc")

        self.tokenizer = Tokenizer(self.data_dir)
        self.vocab_size = self.tokenizer.vocab_size

        # Save the tokenizer vocab to be found by the quantization script
        tokenizer_vocab_path = os.path.join(self.temp_dir, "tokenizer_vocab.json")
        with open(tokenizer_vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.tokenizer.char_to_idx, f)

        self.model_path = os.path.join(self.temp_dir, "model.npz")
        self.quantized_model_path = os.path.join(self.temp_dir, "quantized_model.npz")

    def tearDown(self):
        """Clean up the temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_full_quantization_process(self):
        """
        Tests the full pipeline: save float32, quantize, load int8, and compare outputs.
        """
        # 1. Create and save a float32 model
        transformer_config = TransformerConfig(
            vocab_size=self.vocab_size,
            model=self.config.model,
            vision=self.config.vision,
            ltm=self.config.ltm,
            tokenizer=self.tokenizer
        )
        model_fp32 = Transformer(transformer_config)
        model_fp32.save_weights(self.model_path, self.config_dict)
        self.assertTrue(os.path.exists(self.model_path))

        # 2. Run the quantization script
        with patch('sys.argv', ['', f'--model-path={self.model_path}', f'--output-path={self.quantized_model_path}']):
            quantize_main()
        self.assertTrue(os.path.exists(self.quantized_model_path))

        # 3. Load the quantized model
        model_int8 = Transformer.load_model(
            filepath=self.quantized_model_path,
            vocab_size=self.vocab_size,
            config=self.config,
            tokenizer=self.tokenizer
        )

        # 4. Verify that the weights are indeed quantized
        linear_layer = model_int8.value_head_linear
        self.assertIsNone(linear_layer.weights)
        self.assertIsNotNone(linear_layer.quantized_weights)
        self.assertIsNotNone(linear_layer.weight_scale)
        self.assertEqual(linear_layer.quantized_weights.dtype, np.int8)

        # 5. Compare the outputs of the float32 and int8 models
        dummy_input = np.array([[1, 2, 3]])
        forward_pass_input = ForwardPassInput(x=dummy_input, ltm_state=0)

        # Ensure eval mode is set for consistent dropout behavior (although it's 0)
        model_fp32.eval()
        model_int8.eval()

        logits_fp32, _, _ = model_fp32.forward(forward_pass_input)
        logits_int8, _, _ = model_int8.forward(forward_pass_input)

        # The outputs should be close, but not identical. Due to the small model size
        # and random weights, error accumulation can be significant. A tolerance
        # of ~0.5 is reasonable for this test scenario.
        self.assertTrue(
            np.allclose(logits_fp32, logits_int8, atol=0.5),
            f"Output logits of float32 and quantized int8 models are not close enough. "
            f"Max difference: {np.max(np.abs(logits_fp32 - logits_int8))}"
        )


if __name__ == '__main__':
    unittest.main()
