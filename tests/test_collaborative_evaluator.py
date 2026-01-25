"""
Unit tests for the CollaborativeEvaluator class.
"""
import unittest
from unittest.mock import MagicMock, patch

import torch

from src.agent.agent import Agent
from src.agent.dataclasses import LTMConfig
from src.agent.evaluator import CollaborativeEvaluator
from src.config.core import TrainConfig
from src.model.model import Transformer


class TestCollaborativeEvaluator(unittest.TestCase):
    """Tests for the CollaborativeEvaluator."""

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

        # Create a mock LTMConfig
        self.ltm_config = LTMConfig(
            learning_rate=0.001,
            surprise_threshold=0.1
        )

        self.agents = [
            Agent(self.mock_model, self.ltm_config, agent_id=f"agent_{i}")
            for i in range(3)
        ]

    @patch('src.agent.evaluator.CollaborativeEvaluator._batch_critique')
    def test_collaborative_evaluation_independent_success(self, mock_batch_critique):
        """Test collaboration evaluation for a successful independent response."""
        evaluator = CollaborativeEvaluator(self.agents, self.mock_model)

        # Mock the critique score to be high (successful)
        mock_batch_critique.return_value = 0.9  # Above SUCCESS_THRESHOLD

        # Mock tokenizer and evaluation data
        mock_tokenizer = MagicMock()
        mock_tokenizer.char_to_idx = {"<ASK_FOR_HELP>": 1, "<I_DONT_KNOW>": 2}
        evaluation_data = list(range(100))

        # Mock agent responses to not ask for help or admit ignorance
        for agent in self.agents:
            agent.generate_response = MagicMock(return_value=torch.tensor([[10, 20]]))

        # Run the evaluation
        evaluator.collaborative_evaluation(
            evaluation_data, mock_tokenizer, top_k=3, device='cpu'
        )

        # Check that agents received the correct reward
        expected_reward = evaluator.REWARD_INDEPENDENT_SUCCESS * 0.9
        for agent in self.agents:
            # In this simple case, each agent proposes once
            self.assertAlmostEqual(agent.get_fitness_score(), expected_reward, places=5)

    @patch('src.agent.evaluator.CollaborativeEvaluator._batch_critique')
    def test_collaborative_evaluation_asks_for_help_and_succeeds(self, mock_batch_critique):
        """Test evaluation when an agent asks for help and gets a good response."""
        evaluator = CollaborativeEvaluator(self.agents, self.mock_model)

        # Mock the critique score to be high (successful help)
        mock_batch_critique.return_value = 0.9

        mock_tokenizer = MagicMock()
        mock_tokenizer.char_to_idx = {"<ASK_FOR_HELP>": 1, "<I_DONT_KNOW>": 2}
        evaluation_data = list(range(100))

        # Mock the generate_response method for each agent
        for agent in self.agents:
            agent.generate_response = MagicMock()

        # Agent 0 will ask for help
        self.agents[0].generate_response.return_value = torch.tensor([[1]])
        # Agent 1 will be the helper
        self.agents[1].generate_response.return_value = torch.tensor([[10, 20]])
        # Agent 2 will be a critic
        self.agents[2].generate_response.return_value = torch.tensor([[30, 40]])

        # Run evaluation with just enough data for one proposal (from agent 0)
        prompt_len = self.mock_model.config.model.max_seq_len // 2
        evaluator.collaborative_evaluation(
            evaluation_data[:prompt_len], mock_tokenizer, top_k=3, device='cpu'
        )

        # Agent 0 is the proposer and asks for help
        proposer_score = self.agents[0].get_fitness_score()
        expected_proposer_score = (
            evaluator.REWARD_ASKING_FOR_HELP + (evaluator.REWARD_GOOD_HELP * 0.9)
        )
        self.assertAlmostEqual(proposer_score, expected_proposer_score, places=5)

        # Agent 1 is the helper, and also proposes for itself
        helper_score = self.agents[1].get_fitness_score()
        expected_helper_score = (
            evaluator.REWARD_GOOD_HELP * 0.9 + evaluator.REWARD_INDEPENDENT_SUCCESS * 0.9
        )
        self.assertAlmostEqual(helper_score, expected_helper_score, places=5)

        # Agent 2 also proposes for itself
        critic_score = self.agents[2].get_fitness_score()
        expected_critic_score = evaluator.REWARD_INDEPENDENT_SUCCESS * 0.9
        self.assertAlmostEqual(critic_score, expected_critic_score, places=5)

    def test_batch_critique_handles_ltm_output_tuple(self):
        """
        Tests that _batch_critique correctly unpacks the (context, complexity)
        tuple from the LTM.
        """
        evaluator = CollaborativeEvaluator(self.agents, self.mock_model)

        dummy_sequence = torch.randint(0, 100, (2, 10), device="cpu")
        mock_ltms = [agent.long_term_memory for agent in self.agents[:2]]

        # Mock the LTMs to return a tuple, which was causing the bug
        for ltm in mock_ltms:
            ltm.return_value = (torch.randn(1, 1, 16), torch.tensor([0.5]))

        # Mock the base model's forward pass
        self.mock_model.forward.return_value = (None, torch.tensor([[0.8], [0.7]]), None)
        self.mock_model.layers.embedding.return_value = torch.randn(2, 10, 16)

        # Call the private method. If the bug is present, this will likely
        # raise a TypeError or AttributeError.
        # pylint: disable=protected-access
        avg_score = evaluator._batch_critique(dummy_sequence, mock_ltms)

        # Assert that the score is a valid float, confirming the method ran successfully
        self.assertIsInstance(avg_score, float)
        self.assertAlmostEqual(avg_score, 0.75, places=5)


if __name__ == '__main__':
    unittest.main()
