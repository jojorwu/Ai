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
        logging.error(f"No models found in the '{models_dir}' directory.")
        return None

    available_models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]

    if not available_models:
        logging.error(f"No valid model directories found in '{models_dir}'.")
        return None
    if len(available_models) == 1:
        logging.info(f"Automatically selecting the only available model: {available_models[0]}")
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


def load_model_and_tokenizer(model_name: str, config: Config):
    """Loads the model and tokenizer for a specific model."""
    logging.info(f"Loading model '{model_name}' and tokenizer...")
    model_dir = os.path.join('models', model_name)

    # Prioritize best_model.npz, fall back to model.npz
    weights_path = os.path.join(model_dir, 'best_model.npz')
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_dir, 'model.npz')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"No weights file ('best_model.npz' or 'model.npz') "
                                    f"found in {model_dir}")

    tokenizer = Tokenizer(model_dir)
    model = Transformer.load_model(weights_path, tokenizer.vocab_size, config, tokenizer)
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


def initialize_environment():
    """Initializes the environment, including logging, args, and model selection."""
    setup_logging()
    parser = argparse.ArgumentParser(description="Interact with a trained Transformer model.")
    parser.add_argument('--model-name', type=str, help="The name of the model to use.")
    args = parser.parse_args()

    model_name = args.model_name
    if not model_name:
        model_name = select_model_interactively()
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
    """Runs the main agent loop."""
    start_text = config.generation.start_text
    logging.info(f"Initial task: {start_text}")

    conversation_history_tokens = tokenizer.encode(f"<THINK>{start_text}")

    for turn in range(config.generation.max_turns):
        logging.info(f"\n--- Iteration {turn + 1} ---")

        # The KVCache now handles the sliding window, so we don't need to trim the history here.
        # The model will only 'see' the last `context_window_size` tokens due to the cache.
        gen_config = deepcopy(config.generation)
        generate_input = model.GenerateInput(
            start_tokens=conversation_history_tokens,
            max_new_tokens=gen_config.max_len,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            speculative_steps=gen_config.speculative_steps,
            value_threshold=gen_config.value_threshold,
            max_retries=gen_config.max_retries
        )
        generated_tokens_stream = model.generate(generate_input)
        # We yield from the generator to handle the token stream
        generated_tokens = list(generated_tokens_stream)
        generated_text = tokenizer.decode(generated_tokens)
        logging.info(f"Model generated:\n{generated_text}")

        conversation_history_tokens.extend(tokenizer.encode(generated_text))

        tool_name, args = parse_tool_call(generated_text)
        if tool_name and args is not None:
            tool_output = execute_tool(tool_name, args)
            logging.info(f"Output of tool '{tool_name}':\n{tool_output}")
            tool_output_formatted = f"<TOOL_OUTPUT>{tool_output}</TOOL_OUTPUT>"
            tool_output_tokens = tokenizer.encode(tool_output_formatted)
            conversation_history_tokens.extend(tool_output_tokens)
        else:
            logging.info("\n--- Final Answer ---")
            final_answer = generated_text.split("</TOOL_CALL>")[-1].strip()
            logging.info(final_answer)
            break
    else:
        logging.warning("Maximum number of iterations reached. Terminating.")


def main():
    """Main agent loop."""
    try:
        model, tokenizer, config = initialize_environment()
        if model and tokenizer and config:
            run_agent_loop(model, tokenizer, config)
    except FileNotFoundError as e:
        logging.error("Error: %s. Ensure the model name is correct and the model files exist.", e)
    except (IOError, OSError) as e:
        logging.error("An file system error occurred: %s", e, exc_info=True)
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
