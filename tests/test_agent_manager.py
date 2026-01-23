"""
Unit tests for the AgentManager class.
"""
import unittest
from unittest.mock import MagicMock, patch

import torch
from accelerate import Accelerator

from src.agent.agent_manager import AgentManager
from src.config.core import TrainConfig
from src.model.model import Transformer


class TestAgentManager(unittest.TestCase):
    """Tests for the AgentManager."""

    def setUp(self):
        """Set up a mock model and config for testing."""
        self.mock_config = TrainConfig.from_json('config_train.json')
        self.mock_model = MagicMock(spec=Transformer)
        self.mock_model.config = self.mock_config

        # Create a mock for the 'layers' attribute, which in turn has a 'long_term_memory' attribute
        mock_layers = MagicMock()
        mock_ltm = MagicMock(spec=torch.nn.Module)
        # Make parameters() return a non-empty list to avoid ValueError from Adam optimizer
        mock_ltm.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
        mock_layers.long_term_memory = mock_ltm
        self.mock_model.layers = mock_layers
        self.mock_model.device = 'cpu'

        self.mock_accelerator = MagicMock(spec=Accelerator)
        self.mock_accelerator.unwrap_model.return_value = self.mock_model


    def test_merge_agents_averages_ltm_weights(self):
        """Test that merge_agents correctly averages the LTM weights."""
        manager = AgentManager(
            self.mock_model, num_agents=2, accelerator=self.mock_accelerator
        )

        # Create mock LTM states for two agents
        ltm_state_1 = {'weight': torch.tensor([1.0, 2.0, 3.0])}
        ltm_state_2 = {'weight': torch.tensor([3.0, 4.0, 5.0])}

        manager.agents[0].get_ltm_state = MagicMock(return_value=ltm_state_1)
        manager.agents[1].get_ltm_state = MagicMock(return_value=ltm_state_2)

        # The base model's LTM should be updated with the average
        expected_avg_state = {'weight': torch.tensor([2.0, 3.0, 4.0])}

        # Mock the load_state_dict method to check what it's called with
        self.mock_model.layers.long_term_memory.load_state_dict = MagicMock()

        # Call the method to be tested
        manager.merge_agents(manager.agents)

        # Verify that load_state_dict was called with the correct average
        args, _ = self.mock_model.layers.long_term_memory.load_state_dict.call_args
        loaded_state = args[0]

        self.assertTrue(torch.equal(loaded_state['weight'], expected_avg_state['weight']))


if __name__ == '__main__':
    unittest.main()
