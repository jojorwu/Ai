"""
Main agent script for interacting with the Transformer model using PyTorch.
This script manages the "thought -> tool -> observation" loop,
allowing the model to use tools to complete tasks.
"""
import torch
import logging
import argparse
import os
import re
import json
from dataclasses import dataclass
from typing import List

from accelerate import Accelerator
from config import Config, TransformerConfig
from model import Transformer, GenerateInput
from tokenizer import Tokenizer
from tools import execute_tool
from complexity_manager import ComplexityManager
from utils import setup_logging, select_model_interactively, parse_tool_call

@dataclass
class AgentState:
    """Keeps track of the agent's state during a conversation."""
    conversation_history_tokens: List[int]
    complexity_manager: ComplexityManager = None

def load_model_and_tokenizer(model_name: str, config: Config, load_in_4bit: bool, accelerator: Accelerator):
    """Loads the PyTorch model and tokenizer."""
    logging.info(f"Loading model '{model_name}' (4-bit: {load_in_4bit})...")
    model_dir = os.path.join('models', model_name)
    weights_path = os.path.join(model_dir, 'best_model.pt')
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_dir, 'model.pt')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No weights file found in {model_dir}")

    tokenizer = Tokenizer(model_dir)
    transformer_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size, model=config.model, vision=config.vision,
        ltm=config.ltm, tokenizer=tokenizer
    )

    # Use accelerate's utility to load a quantized model correctly
    model = Transformer(transformer_config, load_in_4bit=load_in_4bit)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)
    model = accelerator.prepare(model)

    model.eval()
    logging.info("Model and tokenizer loaded successfully.")
    return model, tokenizer

def run_agent_loop(model: Transformer, tokenizer: Tokenizer, config: Config, accelerator: Accelerator):
    """Runs the main agent loop."""
    start_text = config.generation.start_text
    logging.info(f"Initial task: {start_text}")

    complexity_manager = ComplexityManager(config.dynamic_parameters) if config.dynamic_parameters else None

    agent_state = AgentState(
        conversation_history_tokens=tokenizer.encode(f"<THINK>{start_text}"),
        complexity_manager=complexity_manager
    )

    for turn in range(config.generation.max_turns):
        logging.info(f"\n--- Iteration {turn + 1} ---")

        input_tokens = torch.tensor([agent_state.conversation_history_tokens], device=accelerator.device)

        dynamic_top_k = agent_state.complexity_manager.get_top_k() if agent_state.complexity_manager else None
        if dynamic_top_k:
             logging.info(f"Complexity: {agent_state.complexity_manager.current_complexity}, Dynamic top_k: {dynamic_top_k}")

        gen_input = GenerateInput(
            start_tokens=input_tokens, max_new_tokens=config.generation.max_len,
            temperature=config.generation.temperature, top_k=config.generation.top_k,
            speculative_steps=config.generation.speculative_steps,
            dynamic_top_k=dynamic_top_k
        )

        newly_generated_tokens = []
        unwrapped_model = accelerator.unwrap_model(model)
        for chunk, surprise in unwrapped_model.generate(gen_input):
            newly_generated_tokens.extend(chunk[0].tolist())
            if agent_state.complexity_manager:
                agent_state.complexity_manager.update_surprise(surprise)

        generated_text = tokenizer.decode(newly_generated_tokens)
        logging.info(f"Model generated:\n{generated_text}")

        agent_state.conversation_history_tokens.extend(newly_generated_tokens)
        full_history_text = tokenizer.decode(agent_state.conversation_history_tokens)

        tool_name, args = parse_tool_call(full_history_text)
        if tool_name and args is not None:
            tool_output = execute_tool(tool_name, args)
            logging.info(f"Output of tool '{tool_name}':\n{tool_output}")

            tool_output_formatted = f"<TOOL_OUTPUT>{tool_output}</TOOL_OUTPUT>"
            agent_state.conversation_history_tokens.extend(tokenizer.encode(tool_output_formatted))
        else:
            logging.info("\n--- Final Answer ---")
            final_answer = generated_text.split("</TOOL_CALL>")[-1].strip()
            print(final_answer)
            break
    else:
        logging.warning("Maximum number of iterations reached.")

def main():
    """Main agent loop for the PyTorch model."""
    setup_logging()
    parser = argparse.ArgumentParser(description="Interact with a PyTorch Transformer model.")
    parser.add_argument('--model-name', type=str, help="The name of the model to use.")
    parser.add_argument('--load-in-4bit', action='store_true', help="Load the model in 4-bit.")
    args = parser.parse_args()

    try:
        model_name = args.model_name or select_model_interactively()
        if not model_name: return

        model_dir = os.path.join('models', model_name)
        config_path = os.path.join(model_dir, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found for model '{model_name}' at {config_path}")

        config = Config.from_json(config_path)
        accelerator = Accelerator()

        model, tokenizer = load_model_and_tokenizer(model_name, config, args.load_in_4bit, accelerator)

        run_agent_loop(model, tokenizer, config, accelerator)

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
