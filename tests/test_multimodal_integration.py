"""
Tests for multimodal integration.
"""
import os
import unittest
import shutil

import numpy as np
from PIL import Image

from config import Config
from model import Transformer
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

        cls.model = Transformer(
            vocab_size=cls.tokenizer.vocab_size,
            model_config=cls.config.model,
            vision_config=cls.config.vision,
            ltm_config=cls.config.ltm,
            tokenizer=cls.tokenizer
        )

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

        logits, value, _ = self.model.forward(token_array, images=image_batch)

        # Expected sequence length = tokens - 1 (for <IMAGE>) + num_patches
        num_patches = (self.config.vision.image_size[0] // self.config.vision.patch_size) ** 2
        expected_seq_len = len(tokens) - 1 + num_patches

        self.assertEqual(logits.shape, (1, expected_seq_len, self.tokenizer.vocab_size))
        self.assertEqual(value.shape, (1, 1))


if __name__ == '__main__':
    unittest.main()
