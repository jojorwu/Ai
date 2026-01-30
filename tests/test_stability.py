"""
Tests for the stability improvements (numerical stability and error handling).
"""
import unittest
import torch
import math
from unittest.mock import MagicMock

from src.agent.agent import Agent
from src.agent.evolution import EvolutionaryOrchestrator
from src.agent.dataclasses import LTMConfig
from src.agent.trainer import AgentTrainer
from src.model.inference.sampling import LogitSampler
from src.model.structures import ForwardOutput
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
        self.mock_model.forward.return_value = ForwardOutput(
            logits=nan_logits,
            value=torch.tensor([[0.5]]),
            aux_loss=torch.tensor(0.1),
            ltm_memory=None
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
        self.mock_model.forward.return_value = ForwardOutput(
            logits=inf_logits,
            value=torch.tensor([[0.5]]),
            aux_loss=torch.tensor(0.1),
            ltm_memory=None
        )

        self.trainer.experience(x, y)
        self.assertEqual(self.agent.metrics.experience_count, 0)

    def test_training_loop_handles_nan_val_loss(self):
        """Tests that TrainingLoop handles NaN validation loss gracefully."""
        mock_trainer = MagicMock()
        mock_trainer.accelerator = MagicMock()
        mock_trainer.run_validation.return_value = float('nan')
        mock_trainer.train_pretrain_epoch.return_value = (0.5, 1.0)

        # Mock callbacks to avoid CheckpointCallback initialization issues
        loop = TrainingLoop(mock_trainer, self.config, callbacks=[MagicMock()])

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

    def test_merge_population_skips_nan_weights(self):
        """Tests that EvolutionaryOrchestrator skips merge if weights contain NaN."""
        # Setup mock agents
        agent_finite = MagicMock(spec=Agent)
        agent_finite.get_fitness_score.return_value = 1.0
        agent_finite.get_ltm_state.return_value = {"w": torch.tensor([1.0])}

        agent_nan = MagicMock(spec=Agent)
        agent_nan.get_fitness_score.return_value = 2.0
        agent_nan.get_ltm_state.return_value = {"w": torch.tensor([float('nan')])}

        # Mock base model
        base_model = MagicMock()
        del base_model.module
        mock_param = MagicMock()
        mock_param.device = torch.device('cpu')
        base_model.layers.long_term_memory.parameters.return_value = iter([mock_param])

        # This should not crash and should log an error (skipping update)
        EvolutionaryOrchestrator.merge_population([agent_finite, agent_nan], base_model)

        # Verify load_state_dict was NOT called because of NaN
        base_model.layers.long_term_memory.load_state_dict.assert_not_called()

    def test_agent_trainer_handles_nan_metrics(self):
        """Tests that AgentTrainer skips metric updates if values are NaN."""
        x = torch.randn(1, 1, self.config.model.d_model)
        y = torch.randint(0, self.config.model.vocab_size, (1, 1))

        # Mock forward to return NaN value but finite logits
        self.mock_model.forward.return_value = ForwardOutput(
            logits=torch.randn(1, 1, self.config.model.vocab_size),
            value=torch.tensor([[float('nan')]]),
            aux_loss=torch.tensor(0.1),
            ltm_memory=None
        )

        initial_count = self.agent.metrics.experience_count
        self.trainer.experience(x, y)

        # Count should not increment
        self.assertEqual(self.agent.metrics.experience_count, initial_count)

if __name__ == "__main__":
    unittest.main()
