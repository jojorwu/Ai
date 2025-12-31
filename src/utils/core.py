"""
Utility functions for the Transformer application.
"""
import json
import logging
import os
import re
from typing import Tuple
import torch
from accelerate import dispatch_model, init_empty_weights


from src.config import Config, TransformerConfig
from src.device_manager import DeviceManager
from src.model import Transformer
from src.tokenizer import Tokenizer

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

def main_entrypoint(main_func):
    """
    Decorator to wrap main functions with common exception handling.
    """

    def wrapper():
        try:
            main_func()
        except FileNotFoundError as e:
            logging.error("File not found: %s", e)
        except (ValueError, TypeError) as e:
            logging.error("Configuration or value error: %s", e)
        except KeyboardInterrupt:
            logging.info("\nExecution interrupted by user.")
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e, exc_info=True)
            raise

    return wrapper

def load_model_and_tokenizer(
    model_name: str,
    config: "Config",
    load_in_4bit: bool,
    quantized: bool,
    dispatch: bool = True,
):
    """Loads a model and tokenizer from a given model name."""
    logging.info(
        "Loading model '%s' (4-bit: %s, quantized: %s)...",
        model_name,
        load_in_4bit,
        quantized,
    )
    model_dir = os.path.join("models", model_name)
    tokenizer = Tokenizer(model_dir)

    transformer_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        model=config.model,
        vision=config.vision,
        ltm=config.ltm,
        tokenizer=tokenizer,
    )
    device_manager = DeviceManager(config.hardware)
    if device_manager.should_disable_4bit():
        if load_in_4bit:
            logging.warning("4-bit quantization is not supported on this hardware, disabling.")
            load_in_4bit = False

    with init_empty_weights():
        model = Transformer(transformer_config, load_in_4bit=load_in_4bit)

    weights_path = os.path.join(model_dir, "best_model.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_dir, "model.pt")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No weights file found in {model_dir}")

    model.load_state_dict(
        torch.load(weights_path, map_location="cpu"), strict=False
    )

    if quantized:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    if dispatch:
        device_map = device_manager.get_device_map()
        model = dispatch_model(model, device_map=device_map)

    model.eval()
    logging.info("Model and tokenizer loaded successfully.")
    return model, tokenizer
