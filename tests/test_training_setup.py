"""
Unit tests for the training setup module.
"""
import argparse
import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os
import shutil

from src.training.setup import prepare_training_environment, TrainingEnvironment


class TestTrainingSetup(unittest.TestCase):
    """Tests for the training setup functions."""

    @patch("src.training.setup.setup_logging")
    @patch("src.training.setup.Accelerator")
    @patch("src.training.setup.load_and_prepare_data")
    def test_prepare_training_environment(
        self,
        mock_load_data,
        mock_accelerator,
        mock_setup_logging,
    ):
        """
        Tests that prepare_training_environment correctly orchestrates the setup.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                shutil.copy(os.path.join(original_cwd, "config_train.json"), "config_train.json")

                args = argparse.Namespace(
                    model_name="test-model",
                    resume_from=None,
                    wandb=False,
                    hardware_strategy=None,
                    torch_compile=False,
                )

                mock_load_data.return_value = (MagicMock(), [1, 2, 3], [4, 5])
                mock_accelerator_instance = MagicMock()
                mock_accelerator.return_value = mock_accelerator_instance

                env = prepare_training_environment(args)

                self.assertIsInstance(env, TrainingEnvironment)
                self.assertTrue(os.path.isdir("models/test-model"))
                self.assertTrue(os.path.isdir("models/test-model/checkpoints"))
                self.assertEqual(env.train_data, [1, 2, 3])
                self.assertEqual(env.val_data, [4, 5])
                self.assertIsNotNone(env.tokenizer)
                self.assertIsNotNone(env.config)
                self.assertEqual(env.accelerator, mock_accelerator_instance)

                mock_load_data.assert_called_once()
                mock_accelerator.assert_called_once()
                mock_setup_logging.assert_called()
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
