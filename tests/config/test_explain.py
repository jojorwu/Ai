"""
Tests for the configuration explanation system.
"""
import unittest
from io import StringIO
from unittest.mock import patch
from src.config.core import BaseConfig, TrainConfig
from tests.test_utils import create_test_config


class TestExplain(unittest.TestCase):
    """Tests for the explanation printer."""

    def test_print_explanations(self):
        """
        Ensures that print_explanations outputs some expected Russian keywords.
        """
        # Using StringIO to capture stdout
        with patch('sys.stdout', new=StringIO()) as fake_out:
            BaseConfig.print_explanations(TrainConfig)
            output = fake_out.getvalue()

            # Check for header
            self.assertIn("Описание настроек", output)
            # Check for some known fields
            self.assertIn("device", output)
            self.assertIn("learning_rate", output)
            # Check for Russian descriptions (keywords)
            self.assertIn("Устройство", output)
            self.assertIn("Скорость", output)


if __name__ == "__main__":
    unittest.main()
