"""
Unit tests for the AgentManager class.
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

import torch
from accelerate import Accelerator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent_manager import AgentManager
from src.config import Config
from src.model import Transformer


class TestAgentManager(unittest.TestCase):
    """Tests for the AgentManager."""

    def setUp(self):
        """Set up a mock model and config for testing."""
        self.mock_config = Config.from_json('config.json')
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

    @patch('src.agent_manager.AgentManager._batch_critique')
    def test_collaborative_evaluation_independent_success(self, mock_batch_critique):
        """Test collaboration evaluation for a successful independent response."""
        manager = AgentManager(
            self.mock_model, num_agents=3, accelerator=self.mock_accelerator
        )

        # Mock the critique score to be high (successful)
        mock_batch_critique.return_value = 0.9  # Above SUCCESS_THRESHOLD

        # Mock tokenizer and evaluation data
        mock_tokenizer = MagicMock()
        mock_tokenizer.char_to_idx = {"<ASK_FOR_HELP>": 1, "<I_DONT_KNOW>": 2}
        evaluation_data = list(range(100))

        # Mock agent responses to not ask for help or admit ignorance
        for agent in manager.agents:
            agent.generate_response = MagicMock(return_value=torch.tensor([[10, 20]]))

        # Run the evaluation
        manager.collaborative_evaluation(
            evaluation_data, mock_tokenizer, top_k=3, device='cpu'
        )

        # Check that agents received the correct reward
        expected_reward = manager.REWARD_INDEPENDENT_SUCCESS * 0.9
        for agent in manager.agents:
            # In this simple case, each agent proposes once
            self.assertAlmostEqual(agent.get_fitness_score(), expected_reward, places=5)

    @patch('src.agent_manager.AgentManager._batch_critique')
    def test_collaborative_evaluation_asks_for_help_and_succeeds(self, mock_batch_critique):
        """Test evaluation when an agent asks for help and gets a good response."""
        manager = AgentManager(
            self.mock_model, num_agents=3, accelerator=self.mock_accelerator
        )

        # Mock the critique score to be high (successful help)
        mock_batch_critique.return_value = 0.9

        mock_tokenizer = MagicMock()
        mock_tokenizer.char_to_idx = {"<ASK_FOR_HELP>": 1, "<I_DONT_KNOW>": 2}
        evaluation_data = list(range(100))

        # Mock the generate_response method for each agent
        for agent in manager.agents:
            agent.generate_response = MagicMock()

        # Agent 0 will ask for help
        manager.agents[0].generate_response.return_value = torch.tensor([[1]])
        # Agent 1 will be the helper
        manager.agents[1].generate_response.return_value = torch.tensor([[10, 20]])
        # Agent 2 will be a critic
        manager.agents[2].generate_response.return_value = torch.tensor([[30, 40]])

        # Run evaluation with just enough data for one proposal (from agent 0)
        prompt_len = self.mock_model.config.model.max_seq_len // 2
        manager.collaborative_evaluation(
            evaluation_data[:prompt_len], mock_tokenizer, top_k=3, device='cpu'
        )

        # Agent 0 is the proposer and asks for help
        proposer_score = manager.agents[0].get_fitness_score()
        expected_proposer_score = (
            manager.REWARD_ASKING_FOR_HELP + (manager.REWARD_GOOD_HELP * 0.9)
        )
        self.assertAlmostEqual(proposer_score, expected_proposer_score, places=5)

        # Agent 1 is the helper, and also proposes for itself
        helper_score = manager.agents[1].get_fitness_score()
        expected_helper_score = (
            manager.REWARD_GOOD_HELP * 0.9 + manager.REWARD_INDEPENDENT_SUCCESS * 0.9
        )
        self.assertAlmostEqual(helper_score, expected_helper_score, places=5)

        # Agent 2 also proposes for itself
        critic_score = manager.agents[2].get_fitness_score()
        expected_critic_score = manager.REWARD_INDEPENDENT_SUCCESS * 0.9
        self.assertAlmostEqual(critic_score, expected_critic_score, places=5)


if __name__ == '__main__':
    unittest.main()
