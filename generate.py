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
from dataclasses import dataclass
from typing import List, Tuple

from backend import np, set_backend
from config import Config, DynamicParametersConfig
from model import Transformer
from tokenizer import Tokenizer
from tools import execute_tool


@dataclass
class AgentState:
    """Keeps track of the agent's state during a conversation."""
    conversation_history_tokens: List[int]
    complexity_manager: 'ComplexityManager' = None


class ComplexityManager:
    """Manages the dynamic complexity level based on a moving average of 'surprise'."""

    def __init__(self, config: DynamicParametersConfig, window_size: int = 5):
        self.config = config
        self.surprise_history = []
        self.window_size = window_size
        self.current_complexity = "low"

    def update_surprise(self, surprise_value: float):
        """Adds a new surprise value and updates the complexity level."""
        self.surprise_history.append(surprise_value)
        if len(self.surprise_history) > self.window_size:
            self.surprise_history.pop(0)
        self._update_complexity()

    def _update_complexity(self):
        """Determines the complexity level based on the average surprise."""
        if not self.surprise_history:
            self.current_complexity = "low"
            return

        avg_surprise = sum(self.surprise_history) / len(self.surprise_history)

        if avg_surprise >= self.config.high_complexity_threshold:
            self.current_complexity = "high"
        elif avg_surprise >= self.config.medium_complexity_threshold:
            self.current_complexity = "medium"
        else:
            self.current_complexity = "low"

    def get_top_k(self) -> int:
        """Returns the top_k value for the current complexity level."""
        if self.current_complexity == "high":
            return self.config.high_complexity_top_k
        if self.current_complexity == "medium":
            return self.config.medium_complexity_top_k
        return self.config.low_complexity_top_k


def setup_logging():
    """Configures console logging."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def select_model_interactively() -> str | None:
    """
    Lists available models and prompts the user to select one.
    """
    models_dir = 'models'
    if not os.path.isdir(models_dir) or not os.listdir(models_dir):
        logging.error("No models found in the '%s' directory.", models_dir)
        return None

    available_models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]

    if not available_models:
        logging.error("No valid model directories found in '%s'.", models_dir)
        return None
    if len(available_models) == 1:
        logging.info("Automatically selecting the only available model: %s", available_models[0])
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


def load_model_and_tokenizer(model_name: str, config: Config) -> Tuple[Transformer, Tokenizer]:
    """Loads the model and tokenizer for a specific model."""
    logging.info("Loading model '%s' and tokenizer...", model_name)
    model_dir = os.path.join('models', model_name)
    weights_path = os.path.join(model_dir, 'best_model.npz')
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_dir, 'model.npz')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"No weights file ('best_model.npz' or 'model.npz') found in {model_dir}"
            )

    tokenizer = Tokenizer(model_dir)
    model = Transformer.load_model(weights_path, tokenizer.vocab_size, config, tokenizer)
    model.eval()
    logging.info("Model and tokenizer loaded successfully.")
    model.quantize_model()
    return model, tokenizer


def parse_tool_call(text: str) -> tuple[str | None, dict | None]:
    """
    Searches for and parses a tool call within <TOOL_CALL> tags.
    """
    pattern = r"<TOOL_CALL>(.*?)</TOOL_CALL>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None, None

    tool_call_json = match.group(1).strip()
    try:
        tool_call = json.loads(tool_call_json)
        tool_name, args = tool_call.get("tool"), tool_call.get("args", {})
        if isinstance(tool_name, str) and isinstance(args, dict):
            return tool_name, args
    except (json.JSONDecodeError, AttributeError):
        pass
    return None, None


def initialize_environment() -> Tuple[Transformer | None, Tokenizer | None, Config | None]:
    """Initializes the environment, including logging, args, and model selection."""
    setup_logging()
    parser = argparse.ArgumentParser(description="Interact with a trained Transformer model.")
    parser.add_argument('--model-name', type=str, help="The name of the model to use.")
    args = parser.parse_args()

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


def _initialize_agent_state(config: Config, tokenizer: Tokenizer) -> AgentState:
    """Initializes the agent's state for a new conversation."""
    start_text = config.generation.start_text
    logging.info("Initial task: %s", start_text)

    complexity_manager = None
    if config.dynamic_parameters:
        complexity_manager = ComplexityManager(config.dynamic_parameters)
        logging.info("Dynamic parameter allocation enabled.")

    return AgentState(
        conversation_history_tokens=tokenizer.encode(f"<THINK>{start_text}"),
        complexity_manager=complexity_manager
    )


def _generate_model_response(model: Transformer, agent_state: AgentState, config: Config) -> Tuple[List[int], str]:
    """Generates a response from the model and updates the complexity manager."""
    dynamic_top_k = None
    if agent_state.complexity_manager:
        dynamic_top_k = agent_state.complexity_manager.get_top_k()
        logging.info("Complexity: %s, Dynamic top_k: %d",
                     agent_state.complexity_manager.current_complexity, dynamic_top_k)

    gen_config = deepcopy(config.generation)
    generate_input = model.GenerateInput(
        start_tokens=agent_state.conversation_history_tokens,
        max_new_tokens=gen_config.max_len,
        temperature=gen_config.temperature,
        top_k=gen_config.top_k,
        top_p=gen_config.top_p,
        speculative_steps=gen_config.speculative_steps,
        value_threshold=gen_config.value_threshold,
        max_retries=gen_config.max_retries,
        dynamic_top_k=dynamic_top_k
    )

    generated_tokens = []
    for chunk, surprise_value in model.generate(generate_input):
        generated_tokens.extend(chunk.tolist())
        if agent_state.complexity_manager:
            agent_state.complexity_manager.update_surprise(surprise_value)

    return generated_tokens


def _process_tool_call(tool_name: str, args: dict, tokenizer: Tokenizer, agent_state: AgentState):
    """Executes a tool call and updates the agent's history."""
    tool_output = execute_tool(tool_name, args)
    logging.info("Output of tool '%s':\n%s", tool_name, tool_output)
    tool_output_formatted = f"<TOOL_OUTPUT>{tool_output}</TOOL_OUTPUT>"
    tool_output_tokens = tokenizer.encode(tool_output_formatted)
    agent_state.conversation_history_tokens.extend(tool_output_tokens)
    return True


def run_agent_loop(model: Transformer, tokenizer: Tokenizer, config: Config):
    """Runs the main agent loop."""
    agent_state = _initialize_agent_state(config, tokenizer)

    for turn in range(config.generation.max_turns):
        logging.info("\n--- Iteration %d ---", turn + 1)

        generated_tokens = _generate_model_response(model, agent_state, config)
        generated_text = tokenizer.decode(generated_tokens)
        logging.info("Model generated:\n%s", generated_text)
        agent_state.conversation_history_tokens.extend(generated_tokens)

        tool_name, args = parse_tool_call(generated_text)
        if tool_name and args is not None:
            _process_tool_call(tool_name, args, tokenizer, agent_state)
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
        logging.error(
            "Error: %s. Ensure the model name is correct and the model files exist.", e)
    except (IOError, OSError) as e:
        logging.error("A file system error occurred: %s", e, exc_info=True)
    except KeyboardInterrupt:
        logging.info("\nProcess interrupted by user. Exiting.")
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
