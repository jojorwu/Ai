"""
Tests for the text generation functionality.
"""
import unittest

from model import GenerateInput
from tests.test_utils import create_test_model


class TestGeneration(unittest.TestCase):
    """
    Tests for the text generation functionality.
    """

    def setUp(self):
        """Set up the test environment."""
        self.model, self.config = create_test_model(ltm=False)

    def test_generate_returns_result(self):
        """
        Tests that the generate method returns a result.
        """
        generate_input = GenerateInput(start_tokens=[1, 2, 3], max_new_tokens=10)
        result = self.model.generate(generate_input)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
