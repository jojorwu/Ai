"""
Tests for multimodal integration.
"""
import os
import shutil
import unittest

import numpy as np
from PIL import Image

from config import Config, TransformerConfig
from model import ForwardPassInput, Transformer
from tokenizer import Tokenizer


class TestMultimodalIntegration(unittest.TestCase):
    """
    Tests for multimodal integration.
    """

    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        cls.temp_dir = "temp_test_dir"
        os.makedirs(cls.temp_dir, exist_ok=True)
        with open(os.path.join(cls.temp_dir, "vocab.txt"), "w", encoding='utf-8') as f:
            f.write("a b c <IMAGE>")

        cls.config = Config.from_json('config.json')
        cls.config.model.d_model = 16
        cls.config.model.num_heads = 2
        cls.config.model.d_ff = 32
        cls.config.model.num_layers = 1
        cls.config.vision.patch_size = 8
        cls.config.vision.image_size = (16, 16)

        cls.tokenizer = Tokenizer(cls.temp_dir)

        transformer_config = TransformerConfig(
            vocab_size=cls.tokenizer.vocab_size,
            model=cls.config.model,
            vision=cls.config.vision,
            ltm=cls.config.ltm,
            tokenizer=cls.tokenizer
        )
        cls.model = Transformer(transformer_config)

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        shutil.rmtree(cls.temp_dir)

    def test_multimodal_forward_pass(self):
        """
        Tests a forward pass with both text and an image.
        """
        text = "a b c <IMAGE>"
        tokens = self.tokenizer.encode(text)
        token_array = np.array([tokens])

        image = Image.new('RGB', self.config.vision.image_size, color='red')
        image_array = np.array(image)
        image_batch = np.array([image_array])

        forward_input = ForwardPassInput(x=token_array, ltm_state=0, images=image_batch)
        logits, value, _ = self.model.forward(forward_input)

        # Expected sequence length = tokens - 1 (for <IMAGE>) + num_patches
        num_patches = (self.config.vision.image_size[0] // self.config.vision.patch_size) ** 2
        expected_seq_len = len(tokens) - 1 + num_patches

        self.assertEqual(logits.shape, (1, expected_seq_len, self.tokenizer.vocab_size))
        self.assertEqual(value.shape, (1, 1))


if __name__ == '__main__':
    unittest.main()
