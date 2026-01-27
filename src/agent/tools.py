"""
Library of tools available for the model to use.
Each tool should be a function with type annotations and a docstring.
"""
import json
import logging
import os
import re
import subprocess
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

class ToolRegistry:
    """
    Manages and executes tools available to the model.
    Encapsulates security measures like safe directory access and whitelisted commands.
    """

    def __init__(self, safe_directory: str = "."):
        self.safe_directory = os.path.abspath(safe_directory)
        self.safe_shell_commands = ["ls", "grep", "echo", "cat", "find", "wc"]
        self.tools = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "create_image": self.create_image,
            "execute_shell": self.execute_shell_command,
        }

    def _is_safe_path(self, path: str) -> bool:
        """Checks if a path is within the safe directory."""
        requested_path = os.path.abspath(os.path.join(self.safe_directory, path))
        return (
            os.path.commonpath([requested_path, self.safe_directory])
            == self.safe_directory
        )

    def list_files(self, path: str = ".") -> str:
        """
        Returns a list of files and directories at the specified path.
        The path must be relative to the project root.
        """
        if not self._is_safe_path(path):
            return "Error: Access outside the working directory is forbidden."

        try:
            files = os.listdir(os.path.join(self.safe_directory, path))
            return json.dumps(files)
        except FileNotFoundError:
            return f"Error: Directory not found at path '{path}'."
        except (PermissionError, OSError) as e:
            return f"Error: An OS error occurred while listing files: {e}"

    def read_file(self, path: str) -> str:
        """
        Reads the content of a file at the specified path.
        The path must be relative to the project root.
        """
        if not self._is_safe_path(path):
            return "Error: Access outside the working directory is forbidden."

        file_path = os.path.join(self.safe_directory, path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: File not found at path '{path}'."
        except (PermissionError, IOError) as e:
            return f"Error: An I/O error occurred while reading the file: {e}"

    def write_file(self, path: str, content: str) -> str:
        """
        Writes the specified content to a file at the given path.
        If the file already exists, it will be overwritten.
        The path must be relative to the project root.
        """
        if not self._is_safe_path(path):
            return "Error: Access outside the working directory is forbidden."

        file_path = os.path.join(self.safe_directory, path)
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Success: File successfully written to '{path}'."
        except (PermissionError, IOError) as e:
            return f"Error: An I/O error occurred while writing the file: {e}"

    def create_image(self, prompt: str, path: str) -> str:
        """
        Creates a simple image with the specified text (prompt) and saves it.
        The path must be relative to the project root and have a .png extension.
        """
        if not self._is_safe_path(path):
            return "Error: Access outside the working directory is forbidden."

        if not path.lower().endswith(".png"):
            return "Error: The image path must end with .png."

        file_path = os.path.join(self.safe_directory, path)
        try:
            img = Image.new("RGB", (400, 200), color=(73, 109, 137))
            drawer = ImageDraw.Draw(img)

            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except IOError:
                font = ImageFont.load_default()

            drawer.text((10, 10), prompt, fill=(255, 255, 0), font=font)

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            img.save(file_path)
            return f"Success: Image created and saved to '{path}'."
        except (IOError, OSError) as e:
            return f"Error: An I/O error occurred while creating the image: {e}"

    def execute_shell_command(self, command: str) -> str:
        """
        Executes a shell command, but only if it is in the approved list of safe commands.
        This is a security measure to prevent the model from executing arbitrary code.
        """
        # Security check: Validate the command against the whitelist.
        command_name = command.strip().split()[0]
        if command_name not in self.safe_shell_commands:
            return (
                f"Error: Command '{command_name}' is not allowed. "
                f"Only the following commands are permitted: {', '.join(self.safe_shell_commands)}"
            )

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,  # 10-second timeout
                check=False,
            )
            if result.returncode == 0:
                return result.stdout
            return f"Error executing command. Exit code: {result.returncode}\nStderr: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 10 seconds."
        except OSError as e:
            return f"An unexpected error occurred: {e}"

    def execute_tool(self, tool_name: str, args: dict) -> str:
        """
        Executes the specified tool with the provided arguments.
        """
        logging.info("Executing tool: %s with args: %s", tool_name, args)
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found."

        tool_function = self.tools[tool_name]
        try:
            return tool_function(**args)
        except TypeError as e:
            return f"Error: Invalid arguments for tool '{tool_name}': {e}"
        except (IOError, OSError) as e:
            return f"A file system error occurred while executing tool '{tool_name}': {e}"


# --- Tool Parsing ---

def parse_tool_call(text: str) -> Tuple[str | None, dict | None]:
    """
    Searches for and parses a tool call within <TOOL_CALL> tags in the given text.
    Returns the tool name and arguments if found, otherwise (None, None).
    """
    pattern = r"<TOOL_CALL>(.*?)</TOOL_CALL>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None, None

    tool_call_json = match.group(1).strip()
    try:
        tool_call = json.loads(tool_call_json)
        tool_name = tool_call.get("tool")
        args = tool_call.get("args", {})
        if isinstance(tool_name, str) and isinstance(args, dict):
            return tool_name, args
    except (json.JSONDecodeError, AttributeError) as e:
        logging.error("Failed to parse tool call: %s\nContent: %s", e, tool_call_json)

    return None, None


