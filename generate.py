"""
Main agent script for interacting with the Transformer model.
This script manages the "thought -> tool -> observation" loop,
allowing the model to use tools to complete tasks.
"""
import json
import logging
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


def load_model_and_tokenizer(config: Config):
    """Loads the model and tokenizer."""
    logging.info("Loading model and tokenizer...")
    tokenizer = Tokenizer(config.evolution.data_dir)
    model = Transformer.load_model(config.evolution.weights_path, tokenizer.vocab_size, config)
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


def main():
    """
    Main agent loop.
    """
    setup_logging()

    try:
        config = Config.from_json('config.json')
        set_backend(config.hardware.device)

        model, tokenizer = load_model_and_tokenizer(config)

        start_text = config.generation.start_text
        logging.info(f"Initial task: {start_text}")

        conversation_history_tokens = tokenizer.encode(f"<THINK>{start_text}")

        for turn in range(config.generation.max_turns):
            logging.info(f"\n--- Iteration {turn + 1} ---")

            gen_config = deepcopy(config.generation)
            gen_config.speculative_steps = 0  # Disable speculation for more precise calls

            generated_tokens_stream = model.generate(
                conversation_history_tokens,
                **gen_config.model_dump()
            )

            generated_text = tokenizer.decode(list(generated_tokens_stream))
            logging.info(f"Model generated:\n{generated_text}")

            conversation_history_tokens.extend(list(generated_tokens_stream))

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
                print(final_answer)
                break
        else:
            logging.warning("Maximum number of iterations reached. Terminating.")

    except FileNotFoundError as e:
        logging.error(f"Error: {e}. Make sure the model is trained and 'config.json' is configured.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
