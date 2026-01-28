"""
Unit tests for the PopulationEvaluator.
"""
import unittest
from unittest.mock import MagicMock, patch

import torch
from src.agent.population_evaluator import PopulationEvaluator


class TestPopulationEvaluator(unittest.TestCase):
    """Tests for the PopulationEvaluator."""

    def test_process_evaluation_data_insufficient(self):
        """Tests that it warns when data is insufficient."""
        evaluator = PopulationEvaluator()

        with self.assertLogs(level='WARNING') as cm:
            evaluator.process_evaluation_data(
                evaluation_data=[1, 2],
                agents=[],
                scores={},
                tokens={},
                device=torch.device("cpu"),
                max_seq_len=100,
                collaborative_evaluator=MagicMock()
            )
            self.assertTrue(any("Not enough validation data" in line for line in cm.output))

    @patch("src.agent.population_evaluator.CollaborationContext")
    def test_evaluate_prompt_logic(self, mock_ctx_cls):
        """Tests the logic for evaluating a single prompt."""
        evaluator = PopulationEvaluator()
        agent = MagicMock()
        # Mock response: first token is 'ask_help'
        agent.generate_response.return_value = torch.tensor([[10, 11]])

        scores = {"agent-1": 0.0}
        tokens = {"ask_help": 10, "i_dont_know": 20}
        agent.agent_id = "agent-1"

        mock_collab_eval = MagicMock()

        evaluator._evaluate_prompt(
            torch.tensor([[1, 2, 3]]), [agent], scores, tokens, mock_collab_eval
        )

        mock_collab_eval.handle_collaboration_request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
