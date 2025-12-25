"""
Main agent script for interacting with the Transformer model.
This script manages the "thought -> tool -> observation" loop,
allowing the model to use tools to complete tasks.
"""
import argparse
import json
import logging
import os
import re
from copy import deepcopy

from backend import set_backend
from config import Config
from model import Transformer
from tokenizer import Tokenizer
from tools import execute_tool


def setup_logging():
    """Configures console logging."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def select_model_interactively() -> str | None:
    """
    Lists available models in the 'models' directory and prompts the user to select one.
    Returns the name of the selected model or None if no models are found.
    """
    models_dir = 'models'
    if not os.path.isdir(models_dir) or not os.listdir(models_dir):
        logging.error("No models found in the '%s' directory.", models_dir)
        return None

    available_models = [d for d in os.listdir(models_dir)
                      if os.path.isdir(os.path.join(models_dir, d))]

    if not available_models:
        logging.error("No valid model directories found in '%s'.", models_dir)
        return None
    if len(available_models) == 1:
        logging.info("Automatically selecting the only available model: %s",
                     available_models[0])
        return available_models[0]

    print("Available models:")
    for i, model_name in enumerate(available_models):
        print(f"  {i + 1}: {model_name}")

    while True:
        try:
            choice = int(input("Please select a model by number: "))
            if 1 <= choice <= len(available_models):
                return available_models[choice - 1]
            print("Invalid number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            print("\nSelection cancelled.")
            return None


def load_model_and_tokenizer(model_name: str, config: Config):
    """Loads the model and tokenizer for a specific model."""
    logging.info("Loading model '%s' and tokenizer...", model_name)
    model_dir = os.path.join('models', model_name)

    # Prioritize best_model.npz, fall back to model.npz
    weights_path = os.path.join(model_dir, 'best_model.npz')
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_dir, 'model.npz')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"No weights file ('best_model.npz' or 'model.npz') "
                                    f"found in {model_dir}")

    tokenizer = Tokenizer(config.evolution.data_dir)
    model = Transformer.load_model(weights_path, tokenizer.vocab_size, config,
                                   tokenizer=tokenizer)
    model.eval()
    logging.info("Model and tokenizer loaded successfully.")
    return model, tokenizer


def parse_tool_call(text: str) -> tuple[str | None, dict | None]:
    """
    Searches for and parses a tool call embedded in the text
    within <TOOL_CALL>...</TOOL_CALL> tags.
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
        if not isinstance(tool_name, str) or not isinstance(args, dict):
            return None, None
        return tool_name, args
    except (json.JSONDecodeError, AttributeError):
        return None, None


def initialize_environment(args):
    """Sets up the model, tokenizer, and configuration."""
    model_name = args.model_name or select_model_interactively()
    if not model_name:
        return None, None, None

    model_dir = os.path.join('models', model_name)
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found for model '{model_name}' at {config_path}")

    config = Config.from_json(config_path)
    set_backend(config.hardware.device)
    model, tokenizer = load_model_and_tokenizer(model_name, config)
    return model, tokenizer, config


def run_agent_loop(model, tokenizer, config):
    """The main loop for the agent's thought-tool-observation cycle."""
    start_text = config.generation.start_text
    logging.info("Initial task: %s", start_text)
    conversation_history = [f"<THINK>{start_text}"]

    for turn in range(config.generation.max_turns):
        logging.info("\n--- Iteration %d ---", turn + 1)

        full_prompt = "".join(conversation_history)
        tokens_to_process = tokenizer.encode(full_prompt)

        context_window = config.generation.context_window_size
        if len(tokens_to_process) > context_window:
            logging.info("Trimming context from %d to %d tokens.",
                         len(tokens_to_process), context_window)
            tokens_to_process = tokens_to_process[-context_window:]

        gen_config = deepcopy(config.generation)
        generated_tokens = list(model.generate(tokens_to_process, **gen_config.model_dump()))
        generated_text = tokenizer.decode(generated_tokens)
        logging.info("Model generated:\n%s", generated_text)
        conversation_history.append(generated_text)

        tool_name, tool_args = parse_tool_call(generated_text)
        if tool_name and tool_args is not None:
            tool_output = execute_tool(tool_name, tool_args)
            logging.info("Output of tool '%s':\n%s", tool_name, tool_output)
            tool_output_formatted = f"<TOOL_OUTPUT>{tool_output}</TOOL_OUTPUT>"
            conversation_history.append(tool_output_formatted)
        else:
            logging.info("\n--- Final Answer ---")
            final_answer = generated_text.rsplit("</TOOL_CALL>", maxsplit=1)[-1].strip()
            print(final_answer)
            return
    logging.warning("Maximum number of iterations reached. Terminating.")


def main():
    """Main agent loop."""
    setup_logging()
    parser = argparse.ArgumentParser(description="Interact with a trained Transformer model.")
    parser.add_argument('--model-name', type=str, help="The name of the model to use.")
    args = parser.parse_args()

    try:
        model, tokenizer, config = initialize_environment(args)
        if model and tokenizer and config:
            run_agent_loop(model, tokenizer, config)
    except FileNotFoundError as e:
        logging.error("Error: %s. Ensure the model name is correct.", e)
    except (json.JSONDecodeError, IOError) as e:
        logging.error("Failed to read or parse config file: %s", e)
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
