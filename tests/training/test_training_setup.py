"""
Unit tests for the training setup module.
"""
import argparse
import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os

from src.training.setup import prepare_training_environment, TrainingEnvironment
from tests.test_utils import create_test_config


class TestTrainingSetup(unittest.TestCase):
    """Tests for the training setup functions."""

    @patch("src.training.setup.setup_logging")
    @patch("src.training.setup.Accelerator")
    @patch("src.training.setup.load_and_prepare_data")
    @patch("src.training.setup.TrainConfig.from_json")
    def test_prepare_training_environment(
        self,
        mock_from_json,
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
                # Mock configuration
                test_config = create_test_config()
                mock_from_json.return_value = test_config

                # Create dummy root config for copying if necessary,
                # but we're mocking from_json anyway.
                os.makedirs("config", exist_ok=True)
                with open("config/config_train.json", "w") as f:
                    f.write("{}")

                args = argparse.Namespace(
                    model_name="test-model",
                    resume_from=None,
                    wandb=False,
                    hardware_strategy=None,
                    torch_compile=False,
                    num_threads=None,
                    num_interop_threads=None,
                    disable_mkldnn=False,
                    flush_denormals=False,
                    num_workers=None,
                    pin_memory=None
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
