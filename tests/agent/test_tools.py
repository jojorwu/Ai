"""
Unit tests for the tools library.
"""
import unittest
from unittest.mock import MagicMock

from src.agent import tools

class TestTools(unittest.TestCase):
    """Tests for the tools."""

    def test_execute_shell_command_safe_command_succeeds(self):
        """Test that a safe, whitelisted command can be executed."""
        command = 'echo "Hello, World!"'
        result = tools.execute_shell_command(command)
        self.assertEqual(result.strip(), "Hello, World!")

    def test_execute_shell_command_unsafe_command_is_blocked(self):
        """Test that a dangerous, non-whitelisted command is blocked."""
        command = "rm -rf /"
        result = tools.execute_shell_command(command)
        self.assertIn("Error: Command 'rm' is not allowed", result)

    def test_execute_tool_nonexistent_tool_fails(self):
        """Test that calling a tool that does not exist returns an error."""
        result = tools.execute_tool("nonexistent_tool", {})
        self.assertIn("Error: Tool 'nonexistent_tool' not found", result)

    def test_execute_tool_shell_command_is_called(self):
        """Test that the execute_tool function correctly calls the shell command tool."""
        original_shell_tool = tools.AVAILABLE_TOOLS['execute_shell']
        tools.AVAILABLE_TOOLS['execute_shell'] = MagicMock(return_value="Success from mock")

        args = {"command": "ls -l"}
        result = tools.execute_tool("execute_shell", args)

        tools.AVAILABLE_TOOLS['execute_shell'].assert_called_once_with(**args)  # pylint: disable=no-member
        self.assertEqual(result, "Success from mock")

        # Restore the original function to avoid side effects in other tests
        tools.AVAILABLE_TOOLS['execute_shell'] = original_shell_tool

if __name__ == '__main__':
    unittest.main()
