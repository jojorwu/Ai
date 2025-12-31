"""
Main agent script for interacting with the Transformer model using PyTorch.
This script manages the "thought -> tool -> observation" loop,
allowing the model to use tools to complete tasks.
"""
import argparse
import logging
import os
from dataclasses import dataclass
from typing import List

import torch
from accelerate import Accelerator, dispatch_model, init_empty_weights

from src.complexity_manager import ComplexityManager
from src.config import Config, TransformerConfig
from src.device_manager import DeviceManager
from src.model import GenerateInput, SamplingConfig, SpeculativeConfig, Transformer
from src.tokenizer import Tokenizer
from src.tools import execute_tool
from src.utils import main_entrypoint, parse_tool_call, select_model_interactively, setup_logging


@dataclass
class AgentState:
    """Keeps track of the agent's state during a conversation."""
    conversation_history_tokens: List[int]
    complexity_manager: ComplexityManager = None

def load_model_and_tokenizer(
    model_name: str,
    config: Config,
    load_in_4bit: bool,
    quantized: bool,
    accelerator: Accelerator,
):
    """Loads the PyTorch model and tokenizer."""
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

    device_map = device_manager.get_device_map()
    model = dispatch_model(model, device_map=device_map)

    model.eval()
    logging.info("Model and tokenizer loaded successfully.")
    return model, tokenizer

def _initialize_agent_state(config, tokenizer):
    """Initializes the agent's state."""
    start_text = config.generation.start_text
    logging.info("Initial task: %s", start_text)
    complexity_manager = ComplexityManager(
        config.dynamic_parameters
    ) if config.dynamic_parameters else None
    return AgentState(
        conversation_history_tokens=tokenizer.encode(f"<THINK>{start_text}"),
        complexity_manager=complexity_manager
    )

def _generate_model_response(model, accelerator, agent_state, config):
    """Generates a response from the model."""
    input_tokens = torch.tensor(
        [agent_state.conversation_history_tokens], device=accelerator.device
    )
    dynamic_top_k = agent_state.complexity_manager.get_top_k(
    ) if agent_state.complexity_manager else None
    if dynamic_top_k:
        logging.info(
            "Complexity: %s, Dynamic top_k: %d",
            agent_state.complexity_manager.current_complexity, dynamic_top_k
        )

    sampling_config = SamplingConfig(
        temperature=config.generation.temperature,
        top_k=config.generation.top_k,
        dynamic_top_k=dynamic_top_k,
    )
    speculative_config = SpeculativeConfig(
        speculative_steps=config.generation.speculative_steps
    )
    gen_input = GenerateInput(
        start_tokens=input_tokens,
        max_new_tokens=config.generation.max_len,
        sampling_config=sampling_config,
        speculative_config=speculative_config,
    )

    newly_generated_tokens = []
    unwrapped_model = accelerator.unwrap_model(model)
    for chunk, surprise in unwrapped_model.generate(gen_input):
        newly_generated_tokens.extend(chunk[0].tolist())
        if agent_state.complexity_manager:
            agent_state.complexity_manager.update_surprise(surprise)
    return newly_generated_tokens

def _process_tool_call(agent_state, tokenizer):
    """Processes a tool call if one is present in the conversation history."""
    full_history_text = tokenizer.decode(agent_state.conversation_history_tokens)
    tool_name, args = parse_tool_call(full_history_text)
    if tool_name and args is not None:
        tool_output = execute_tool(tool_name, args)
        logging.info("Output of tool '%s':\n%s", tool_name, tool_output)
        tool_output_formatted = f"<TOOL_OUTPUT>{tool_output}</TOOL_OUTPUT>"
        agent_state.conversation_history_tokens.extend(
            tokenizer.encode(tool_output_formatted)
        )
        return True
    return False

def run_agent_loop(
    model: Transformer, tokenizer: Tokenizer, config: Config, accelerator: Accelerator
):
    """Runs the main agent loop."""
    agent_state = _initialize_agent_state(config, tokenizer)

    for turn in range(config.generation.max_turns):
        logging.info("\n--- Iteration %d ---", turn + 1)
        newly_generated_tokens = _generate_model_response(
            model, accelerator, agent_state, config
        )
        generated_text = tokenizer.decode(newly_generated_tokens)
        logging.info("Model generated:\n%s", generated_text)
        agent_state.conversation_history_tokens.extend(newly_generated_tokens)

        if not _process_tool_call(agent_state, tokenizer):
            logging.info("\n--- Final Answer ---")
            final_answer = generated_text.split("</TOOL_CALL>")[-1].strip()
            print(final_answer)
            break
    else:
        logging.warning("Maximum number of iterations reached.")

@main_entrypoint
def main():
    """Main agent loop for the PyTorch model."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Interact with a PyTorch Transformer model.")
    parser.add_argument(
        '--model-name', type=str, help="The name of the model to use.")
    parser.add_argument(
        '--load-in-4bit', action='store_true', help="Load the model in 4-bit.")
    parser.add_argument(
        '--quantized', action='store_true', help="Load a quantized model for CPU."
    )
    args = parser.parse_args()

    model_name = args.model_name or select_model_interactively()
    if not model_name:
        return

    model_dir = os.path.join('models', model_name)
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found for model '{model_name}' at {config_path}"
        )

    config = Config.from_json(config_path)
    accelerator = Accelerator()
    model, tokenizer = load_model_and_tokenizer(
        model_name, config, args.load_in_4bit, args.quantized, accelerator
    )
    run_agent_loop(model, tokenizer, config, accelerator)

if __name__ == "__main__":
    main()
