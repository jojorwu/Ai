"""
Tests for the main agent execution loop in generate.py.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We need to import the main function from the script
import generate

class TestAgentExecution(unittest.TestCase):

    @patch('generate.load_model_and_tokenizer')
    @patch('generate.Config.from_json')
    def test_agent_loop_terminates_on_no_tool_call(self, mock_config, mock_load_model):
        """
        Tests that the agent loop correctly identifies a final answer (no tool call)
        and terminates.
        """
        # --- Mock Setup ---
        # Mock the config to control the loop
        mock_config.return_value.generation.max_turns = 5
        mock_config.return_value.generation.start_text = "Initial prompt"
        # Set a valid device for the backend
        mock_config.return_value.hardware.device = "cpu"

        # Mock the tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "This is the final answer."

        # Mock the model
        mock_model = MagicMock()
        # The first call to generate should produce a response without a tool call
        mock_model.generate.return_value = ([4, 5, 6], 0.0, 0.0)

        # load_model_and_tokenizer should return our mocks
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # --- Test Execution ---
        # We run the main function from the generate script
        generate.main()

        # --- Assertions ---
        # The model's generate method should have been called exactly once
        mock_model.generate.assert_called_once()

        # The loop should terminate after the first turn because no tool call was found
        # (This is implicitly tested by the fact that generate was only called once)


if __name__ == '__main__':
    unittest.main()
