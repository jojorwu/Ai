"""
Tests for TrainingState preservation and serialization.
"""
import unittest
import torch
import tempfile
import os
from src.training.loop import TrainingState
from accelerate import Accelerator

class TestStatePreservation(unittest.TestCase):
    def test_training_state_serialization(self):
        """Verifies that TrainingState correctly serializes and restores its fields."""
        state = TrainingState(epoch=5, best_val_loss=0.123, epochs_no_improve=2)

        sd = state.state_dict()
        self.assertEqual(sd["epoch"], 5)
        self.assertEqual(sd["best_val_loss"], 0.123)
        self.assertEqual(sd["epochs_no_improve"], 2)

        new_state = TrainingState()
        new_state.load_state_dict(sd)
        self.assertEqual(new_state.epoch, 5)
        self.assertEqual(new_state.best_val_loss, 0.123)
        self.assertEqual(new_state.epochs_no_improve, 2)

    def test_accelerator_checkpointing_integration(self):
        """Verifies TrainingState integration with Accelerator checkpointing."""
        accelerator = Accelerator()
        state = TrainingState(epoch=10, best_val_loss=0.5, epochs_no_improve=1)
        accelerator.register_for_checkpointing(state)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save state
            accelerator.save_state(tmpdir)

            # Modify state
            state.epoch = 0
            state.best_val_loss = 1.0
            state.epochs_no_improve = 0

            # Load state
            accelerator.load_state(tmpdir)

            # Verify restoration
            self.assertEqual(state.epoch, 10)
            self.assertEqual(state.best_val_loss, 0.5)
            self.assertEqual(state.epochs_no_improve, 1)

if __name__ == "__main__":
    unittest.main()
