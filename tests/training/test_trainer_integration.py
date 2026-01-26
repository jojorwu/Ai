"""
Integration test for the PyTorch-based Trainer class.
"""
import unittest

import torch
from accelerate import Accelerator

from src.config.core import TrainConfig
from src.training.trainer import create_trainer
from tests.test_utils import create_full_test_config_and_data, MockTokenizer


class TestTrainerIntegration(unittest.TestCase):  # pylint: disable=duplicate-code
    """
    Tests the Trainer's ability to run pre-training and evolution cycles.
    """

    def test_trainer_runs_and_updates_model(self):
        """
        Tests that a pre-training epoch and an evolution cycle both update the base model.
        """
        config, tokenizer, train_data, val_data = create_full_test_config_and_data()
        accelerator = Accelerator()

        trainer = create_trainer(
            config=config,
            tokenizer=MockTokenizer(),
            train_data=train_data,
            val_data=val_data,
            accelerator=accelerator,
        )
        model = trainer.get_model()
        unwrapped_model = model.module if hasattr(model, "module") else model

        # --- 1. Test Pre-training ---
        initial_weights_pre = (
            unwrapped_model.layers.decoder[0].attention_sublayer.mha.wo.weight.clone().detach()
        )

        trainer.train_pretrain_epoch()

        updated_weights_pre = (
            unwrapped_model.layers.decoder[0].attention_sublayer.mha.wo.weight.clone().detach()
        )

        self.assertFalse(
            torch.equal(initial_weights_pre, updated_weights_pre),
            "Model weights were not updated after a pre-training epoch."
        )

        # --- 2. Test Evolution Cycle ---
        trainer.run_evolution_cycle()

if __name__ == "__main__":
    unittest.main()
