"""
Library of tools available for the model to use.
Each tool should be a function with type annotations and a docstring.
"""
import json
import logging
import os

from PIL import Image, ImageDraw, ImageFont

# --- Security: Define the allowed working directory ---
# This is the project's root directory. Tools cannot go outside of it.
SAFE_DIRECTORY = os.path.abspath(".")


def _is_safe_path(path: str) -> bool:
    """Checks if a path is within the SAFE_DIRECTORY."""
    requested_path = os.path.abspath(os.path.join(SAFE_DIRECTORY, path))
    return os.path.commonpath([requested_path, SAFE_DIRECTORY]) == SAFE_DIRECTORY


# --- Filesystem Tools ---

def list_files(path: str = ".") -> str:
    """
    Returns a list of files and directories at the specified path.
    The path must be relative to the project root.
    """
    if not _is_safe_path(path):
        return "Error: Access outside the working directory is forbidden."

    try:
        files = os.listdir(os.path.join(SAFE_DIRECTORY, path))
        return json.dumps(files)
    except FileNotFoundError:
        return f"Error: Directory not found at path '{path}'."
    except Exception as e:
        return f"Error: An error occurred while listing files: {e}"


def read_file(path: str) -> str:
    """
    Reads the content of a file at the specified path.
    The path must be relative to the project root.
    """
    if not _is_safe_path(path):
        return "Error: Access outside the working directory is forbidden."

    file_path = os.path.join(SAFE_DIRECTORY, path)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found at path '{path}'."
    except Exception as e:
        return f"Error: An error occurred while reading the file: {e}"


def write_file(path: str, content: str) -> str:
    """
    Writes the specified content to a file at the given path.
    If the file already exists, it will be overwritten.
    The path must be relative to the project root.
    """
    if not _is_safe_path(path):
        return "Error: Access outside the working directory is forbidden."

    file_path = os.path.join(SAFE_DIRECTORY, path)
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Success: File successfully written to '{path}'."
    except Exception as e:
        return f"Error: An error occurred while writing the file: {e}"


# --- Image Generation Tool ---

def create_image(prompt: str, path: str) -> str:
    """
    Creates a simple image with the specified text (prompt) and saves it.
    The path must be relative to the project root and have a .png extension.
    """
    if not _is_safe_path(path):
        return "Error: Access outside the working directory is forbidden."

    if not path.lower().endswith('.png'):
        return "Error: The image path must end with .png."

    file_path = os.path.join(SAFE_DIRECTORY, path)
    try:
        img = Image.new('RGB', (400, 200), color=(73, 109, 137))
        drawer = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        drawer.text((10, 10), prompt, fill=(255, 255, 0), font=font)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        img.save(file_path)
        return f"Success: Image created and saved to '{path}'."
    except Exception as e:
        return f"Error: An error occurred while creating the image: {e}"


# --- Tool Registry ---

AVAILABLE_TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "create_image": create_image,
}


# pylint: disable=broad-except-in-catch
def execute_tool(tool_name: str, args: dict) -> str:
    """
    Executes the specified tool with the provided arguments.
    """
    logging.info(f"Executing tool: {tool_name} with args: {args}")
    if tool_name not in AVAILABLE_TOOLS:
        return f"Error: Tool '{tool_name}' not found."

    tool_function = AVAILABLE_TOOLS[tool_name]
    try:
        return tool_function(**args)
    except TypeError as e:
        return f"Error: Invalid arguments for tool '{tool_name}': {e}"
    except Exception as e:
        return f"Error: An unexpected error occurred while executing tool '{tool_name}': {e}"
