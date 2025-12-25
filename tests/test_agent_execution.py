"""
Tests for the main agent execution loop in generate.py.
"""
import unittest
from unittest.mock import MagicMock, Mock, patch

import generate


class TestAgentExecution(unittest.TestCase):
    """
    Tests for the main agent execution loop.
    """

    @patch('generate.Config.from_json')
    @patch('generate.load_model_and_tokenizer')
    @patch('os.path.exists', return_value=True)
    @patch('argparse.ArgumentParser.parse_args')
    def test_agent_loop_terminates_on_no_tool_call(
            self, mock_parse_args, _, mock_load_model, mock_config):
        """
        Tests that the agent loop correctly identifies a final answer
        (no tool call) and terminates.
        """
        # Mock command-line arguments to prevent conflict with the test runner
        mock_parse_args.return_value = Mock(model_name='test-model')

        # Mock config and model loading
        mock_config.return_value.generation.max_turns = 5
        mock_config.return_value.generation.start_text = "Initial prompt"
        mock_config.return_value.generation.context_window_size = 1024  # Fix TypeError
        mock_config.return_value.hardware.device = "cpu"

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "This is the final answer."

        mock_model = MagicMock()
        mock_model.generate.return_value = iter([4, 5, 6])

        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # Run the main function from the generate script
        generate.main()

        # Assert that the model's generate method was called exactly once
        mock_model.generate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
