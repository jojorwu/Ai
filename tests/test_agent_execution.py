"""
Tests for the main agent execution loop in generate.py.
"""
import unittest
from unittest.mock import MagicMock, patch

import generate


class TestAgentExecution(unittest.TestCase):
    """
    Tests for the main agent execution loop.
    """

    @patch('generate.load_model_and_tokenizer')
    @patch('generate.Config.from_json')
    def test_agent_loop_terminates_on_no_tool_call(self, mock_config, mock_load_model):
        """
        Tests that the agent loop correctly identifies a final answer
        (no tool call) and terminates.
        """
        mock_config.return_value.generation.max_turns = 5
        mock_config.return_value.generation.start_text = "Initial prompt"
        mock_config.return_value.hardware.device = "cpu"

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "This is the final answer."

        mock_model = MagicMock()
        mock_model.generate.return_value = [4, 5, 6]

        mock_load_model.return_value = (mock_model, mock_tokenizer)

        generate.main()

        mock_model.generate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
