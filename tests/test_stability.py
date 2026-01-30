"""
Tests for the stability improvements (numerical stability and error handling).
"""
import unittest
import torch
import math
from unittest.mock import MagicMock

from src.agent.agent import Agent
from src.agent.dataclasses import LTMConfig
from src.agent.trainer import AgentTrainer
from src.model.inference.sampling import LogitSampler
from src.training.loop import TrainingLoop, TrainingState
from tests.test_utils import create_test_config

class TestStability(unittest.TestCase):
    """Verifies that the system handles numerical issues gracefully."""

    def setUp(self):
        self.config = create_test_config()
        self.ltm_config = LTMConfig(learning_rate=0.01, surprise_threshold=0.1)

        # Mock model and layers
        self.mock_model = MagicMock()
        self.mock_model.config = self.config
        self.mock_model.layers = MagicMock()

        # Mock LTM
        self.mock_ltm = MagicMock(spec=torch.nn.Module)
        self.param = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.mock_ltm.parameters.return_value = [self.param]
        self.mock_model.layers.long_term_memory = self.mock_ltm

        self.agent = Agent(self.mock_model, self.ltm_config)
        self.trainer = AgentTrainer(self.agent)

    def test_agent_trainer_handles_nan_loss(self):
        """Tests that AgentTrainer skips update if loss is NaN."""
        x = torch.randn(1, 1, self.config.model.d_model)
        y = torch.randint(0, self.config.model.vocab_size, (1, 1))

        # Mock forward to return NaN
        nan_logits = torch.full((1, 1, self.config.model.vocab_size), float('nan'))
        self.mock_model.forward.return_value = (
            nan_logits,          # logits
            torch.tensor([[0.5]]),           # values
            torch.tensor(0.1),               # aux_loss
            None                             # ltm_memory
        )

        # This should not raise an exception and should skip optimizer.step()
        self.trainer.experience(x, y)

        # If experience() returned early, it didn't call _update_ltm_and_calc_surprise
        self.assertEqual(self.agent.metrics.experience_count, 0)

    def test_agent_trainer_handles_inf_loss(self):
        """Tests that AgentTrainer skips update if loss is Inf."""
        x = torch.randn(1, 1, self.config.model.d_model)
        y = torch.randint(0, self.config.model.vocab_size, (1, 1))

        # Mock forward to return Inf
        inf_logits = torch.full((1, 1, self.config.model.vocab_size), float('inf'))
        self.mock_model.forward.return_value = (
            inf_logits,          # logits
            torch.tensor([[0.5]]),           # values
            torch.tensor(0.1),               # aux_loss
            None                             # ltm_memory
        )

        self.trainer.experience(x, y)
        self.assertEqual(self.agent.metrics.experience_count, 0)

    def test_training_loop_handles_nan_val_loss(self):
        """Tests that TrainingLoop handles NaN validation loss gracefully."""
        mock_trainer = MagicMock()
        mock_trainer.accelerator = MagicMock()
        mock_trainer.run_validation.return_value = float('nan')
        mock_trainer.train_pretrain_epoch.return_value = (0.5, 1.0)

        loop = TrainingLoop(mock_trainer, self.config)

        # Force it to run only 1 epoch
        self.config.evolution.pretrain_epochs = 1
        self.config.evolution.evolution_epochs = 0

        # This should not raise an exception and should not save state
        loop.run("dummy_dir", None)

        # Verify save_state was NOT called because val_loss was NaN
        mock_trainer.accelerator.save_state.assert_not_called()

    def test_logit_sampler_handles_nan_logits(self):
        """Tests that LogitSampler falls back to greedy if logits are all NaN/filtered."""
        sampler = LogitSampler()
        # All logits are -inf after filtering or NaN originally
        logits = torch.full((1, 10), float('nan'))

        # This should fallback to greedy and return an index (not crash)
        token = sampler.sample(logits, temperature=1.0)
        self.assertEqual(token.shape, (1, 1))
        self.assertTrue(0 <= token.item() < 10)

if __name__ == "__main__":
    unittest.main()
