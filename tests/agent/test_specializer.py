"""
Unit tests for the AgentSpecializer.
"""
import unittest
from unittest.mock import MagicMock, patch

import torch
from src.agent.specializer import AgentSpecializer
from src.agent.dataclasses import SpecializationConfig


class TestSpecializer(unittest.TestCase):
    """Tests for the AgentSpecializer."""

    def test_specialize_agents_empty_data(self):
        """Tests that it handles empty data gracefully."""
        specializer = AgentSpecializer()
        agents = [MagicMock()]
        config = SpecializationConfig(
            full_data=[], seq_len=10, batch_size=1, steps_per_agent=5
        )

        with self.assertLogs(level='WARNING') as cm:
            specializer.specialize_agents(agents, config, torch.device("cpu"))
            self.assertTrue(any("No data provided" in line for line in cm.output))

    @patch("src.agent.specializer.AgentTrainer")
    @patch("src.agent.specializer.get_batches_torch")
    def test_specialize_agents_logic(self, mock_get_batches, mock_trainer_cls):
        """Tests the orchestration of agent specialization."""
        specializer = AgentSpecializer()
        agent = MagicMock()
        agent.agent_id = "agent-1"
        agents = [agent]

        config = SpecializationConfig(
            full_data=list(range(100)),
            seq_len=10,
            batch_size=2,
            steps_per_agent=2
        )

        # Mock batch generator
        mock_get_batches.return_value = [
            (torch.tensor([1]), torch.tensor([2]), None),
            (torch.tensor([3]), torch.tensor([4]), None),
            (torch.tensor([5]), torch.tensor([6]), None),
        ]

        mock_trainer = mock_trainer_cls.return_value

        specializer.specialize_agents(agents, config, torch.device("cpu"))

        self.assertEqual(mock_trainer.experience.call_count, 2)
        mock_get_batches.assert_called_once()


if __name__ == "__main__":
    unittest.main()
