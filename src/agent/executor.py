"""
This module contains the AgentExecutor class, which is responsible for
running the main agent loop.
"""
import logging
from dataclasses import dataclass
from typing import List

import torch
from accelerate import Accelerator

from src.agent.tools import execute_tool, parse_tool_call
from src.config.core import GenerateConfig
from src.data.tokenizer import Tokenizer
from src.model.complexity_manager import ComplexityManager
from src.model.model import (
    GenerateInput,
    SamplingConfig,
    SpeculativeConfig,
    Transformer,
)


class AgentExecutor:
    @dataclass
    class AgentState:
        """Keeps track of the agent's state during a conversation."""
        conversation_history_tokens: List[int]
        complexity_manager: ComplexityManager = None
    """
    Handles the main execution loop of the agent, including the
    "thought -> tool -> observation" cycle.
    """

    def __init__(
        self,
        model: Transformer,
        tokenizer: Tokenizer,
        config: GenerateConfig,
        accelerator: Accelerator,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.accelerator = accelerator

    def run(self):
        """Runs the main agent loop."""
        agent_state = self._initialize_agent_state()

        for turn in range(self.config.generation.max_turns):
            logging.info("\n--- Iteration %d ---", turn + 1)
            newly_generated_tokens = self._generate_model_response(agent_state)
            generated_text = self.tokenizer.decode(newly_generated_tokens)
            logging.info("Model generated:\n%s", generated_text)
            agent_state.conversation_history_tokens.extend(newly_generated_tokens)

            if not self._process_tool_call(agent_state):
                logging.info("\n--- Final Answer ---")
                final_answer = generated_text.split("</TOOL_CALL>")[-1].strip()
                print(final_answer)
                break
        else:
            logging.warning("Maximum number of iterations reached.")

    def _initialize_agent_state(self) -> AgentState:
        """Initializes the agent's state."""
        start_text = self.config.generation.start_text
        logging.info("Initial task: %s", start_text)
        complexity_manager = (
            ComplexityManager(self.config.dynamic_parameters)
            if self.config.dynamic_parameters
            else None
        )
        return AgentState(
            conversation_history_tokens=self.tokenizer.encode(f"<THINK>{start_text}"),
            complexity_manager=complexity_manager,
        )

    def _generate_model_response(self, agent_state: AgentState) -> List[int]:
        """Generates a response from the model."""
        input_tokens = torch.tensor(
            [agent_state.conversation_history_tokens], device=self.accelerator.device
        )
        dynamic_top_k = (
            agent_state.complexity_manager.get_top_k()
            if agent_state.complexity_manager
            else None
        )
        if dynamic_top_k:
            logging.info(
                "Complexity: %s, Dynamic top_k: %d",
                agent_state.complexity_manager.current_complexity,
                dynamic_top_k,
            )

        sampling_config = SamplingConfig(
            temperature=self.config.generation.temperature,
            top_k=self.config.generation.top_k,
            dynamic_top_k=dynamic_top_k,
        )
        speculative_config = SpeculativeConfig(
            speculative_steps=self.config.generation.speculative_steps
        )
        gen_input = GenerateInput(
            start_tokens=input_tokens,
            max_new_tokens=self.config.generation.max_len,
            sampling_config=sampling_config,
            speculative_config=speculative_config,
        )

        newly_generated_tokens = []
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        for chunk, surprise in unwrapped_model.generate(gen_input):
            newly_generated_tokens.extend(chunk[0].tolist())
            if agent_state.complexity_manager:
                agent_state.complexity_manager.update_surprise(surprise)
        return newly_generated_tokens

    def _process_tool_call(self, agent_state: AgentState) -> bool:
        """Processes a tool call if one is present in the conversation history."""
        full_history_text = self.tokenizer.decode(
            agent_state.conversation_history_tokens
        )
        tool_name, args = parse_tool_call(full_history_text)
        if tool_name and args is not None:
            tool_output = execute_tool(tool_name, args)
            logging.info("Output of tool '%s':\n%s", tool_name, tool_output)
            tool_output_formatted = f"<TOOL_OUTPUT>{tool_output}</TOOL_OUTPUT>"
            agent_state.conversation_history_tokens.extend(
                self.tokenizer.encode(tool_output_formatted)
            )
            return True
        return False
