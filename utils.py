"""
Utility functions for the Transformer application.
"""
import json
import logging
import os
import re
from typing import Tuple

def setup_logging(log_path: str = None):
    """
    Configures logging to file and console.
    If log_path is None, only logs to console.
    """
    handlers = [logging.StreamHandler()]
    if log_path:
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers
    )
    logging.getLogger().setLevel(logging.INFO)

def select_model_interactively() -> str | None:
    """
    Lists available models in the 'models' directory and prompts the user to select one.
    Returns the selected model name or None if no selection is made.
    """
    models_dir = 'models'
    if not os.path.isdir(models_dir) or not os.listdir(models_dir):
        logging.error("No models found in the '%s' directory.", models_dir)
        return None

    available_models = [
        d for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d))
    ]

    if not available_models:
        logging.error("No valid model directories found in '%s'.", models_dir)
        return None
    if len(available_models) == 1:
        logging.info("Automatically selecting the only available model: %s",
                     available_models[0])
        return available_models[0]

    logging.info("Available models:")
    for i, model_name in enumerate(available_models):
        logging.info("  %d: %s", i + 1, model_name)

    while True:
        try:
            choice = int(input("Please select a model by number: "))
            if 1 <= choice <= len(available_models):
                return available_models[choice - 1]
            logging.warning("Invalid number. Please try again.")
        except ValueError:
            logging.warning("Invalid input. Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            logging.info("\nSelection cancelled.")
            return None

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
