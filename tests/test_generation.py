"""
Tests for the text generation functionality.
"""
import unittest

from config import Config
from model import Transformer


class TestGeneration(unittest.TestCase):
    """
    Tests for the text generation functionality.
    """

    def setUp(self):
        """Set up the test environment."""
        self.config = Config.from_json('config.json')
        self.model = Transformer(
            vocab_size=50,
            model_config=self.config.model,
            vision_config=self.config.vision,
            ltm_config=self.config.ltm
        )

    def test_generate_returns_result(self):
        """
        Tests that the generate method returns a result.
        """
        result = self.model.generate(start_tokens=[1, 2, 3], max_new_tokens=10)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
