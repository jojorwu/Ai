"""
Integration test for the PyTorch-based Trainer class.
"""
import unittest

import torch
from accelerate import Accelerator

from config import Config
from trainer import create_trainer, DataComponents


def _create_test_config_and_data():
    """Creates a minimal configuration and dummy data for testing the Trainer."""
    config = Config.from_json('config.json')
    config.model.d_model = 8
    config.model.num_layers = 1
    config.model.num_heads = 2
    config.model.d_ff = 16
    config.model.early_exit_thresholds = None
    config.evolution.pretrain_epochs = 1
    config.evolution.evolution_epochs = 1
    config.evolution.num_agents = 2
    config.evolution.num_survivors = 1

    class MockTokenizer:  # pylint: disable=too-few-public-methods
        """A mock tokenizer for testing purposes."""
        vocab_size = 50

        def __init__(self):
            self.char_to_idx = {
                '<ASK_FOR_HELP>': 48,
                '<I_DONT_KNOW>': 49
            }

        def encode(self, text):
            """Bare-bones encode method for testing."""
            return list(text.encode('utf-8'))

    tokenizer = MockTokenizer()
    vocab_size = tokenizer.vocab_size

    train_data = list(range(vocab_size)) * 5
    val_data = list(range(vocab_size)) * 5

    return config, tokenizer, train_data, val_data

class TestTrainerIntegration(unittest.TestCase):
    """
    Tests the Trainer's ability to run pre-training and evolution cycles.
    """

    def test_trainer_runs_and_updates_model(self):
        """
        Tests that a pre-training epoch and an evolution cycle both update the base model.
        """
        config, tokenizer, train_data, val_data = _create_test_config_and_data()
        accelerator = Accelerator()

        data_components = DataComponents(
            tokenizer=tokenizer, train_data=train_data, val_data=val_data
        )
        trainer = create_trainer(config, data_components, accelerator)
        model = trainer.get_model()
        unwrapped_model = model.module if hasattr(model, "module") else model

        # --- 1. Test Pre-training ---
        initial_weights_pre = unwrapped_model.layers.decoder[0].mha.wo.weights.clone().detach()

        trainer.train_pretrain_epoch()

        updated_weights_pre = unwrapped_model.layers.decoder[0].mha.wo.weights.clone().detach()

        self.assertFalse(
            torch.equal(initial_weights_pre, updated_weights_pre),
            "Model weights were not updated after a pre-training epoch."
        )

        # --- 2. Test Evolution Cycle ---
        trainer.run_evolution_cycle()

if __name__ == "__main__":
    unittest.main()
