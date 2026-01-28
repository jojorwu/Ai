"""
This module contains the AgentExecutor class, which is responsible for
running the main agent loop.
"""
import logging
from dataclasses import dataclass
from typing import List

import torch
from accelerate import Accelerator

from src.agent.cache_manager import CacheManager
from src.agent.dataclasses import AgentState
from src.agent.tools import ToolRegistry, parse_tool_call
from src.config.core import GenerateConfig
from src.data.tokenizer import Tokenizer
from src.model.complexity_manager import ComplexityManager
from src.model.model import Transformer
from src.model.structures import (
    GenerateInput,
    SamplingConfig,
    SpeculativeConfig,
)


class AgentExecutor:
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
        self.tool_registry = ToolRegistry()
        self.cache_manager = CacheManager(accelerator)

    def run(self):
        """Runs the main agent loop."""
        agent_state = self._initialize_agent_state()

        for turn in range(self.config.generation.max_turns):
            logging.info("\n--- Iteration %d ---", turn + 1)

            # Generate model response
            new_tokens = self._generate_model_response(agent_state)
            agent_state.append_tokens(new_tokens)
            agent_state.prune_history(self.config.generation.context_window_size)

            generated_text = self.tokenizer.decode(new_tokens)
            logging.info("Model generated:\n%s", generated_text)

            # Check for tool calls
            if not self._process_tool_call(agent_state):
                logging.info("\n--- Final Answer ---")
                # Attempt to extract the final answer from the last generated text
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
            ComplexityManager(self.config.complexity)
            if self.config.complexity
            else None
        )
        return AgentState(
            conversation_history_tokens=self.tokenizer.encode(f"<THINK>{start_text}"),
            complexity_manager=complexity_manager,
        )

    def _generate_model_response(self, agent_state: AgentState) -> List[int]:
        """Generates a response from the model, utilizing persistent KV caching."""
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Initialize KV Caches if they don't exist yet.
        if agent_state.main_cache is None:
            self.cache_manager.initialize_agent_caches(agent_state, unwrapped_model)

        # Use new_tokens from AgentState
        new_tokens = agent_state.get_new_tokens()
        if not new_tokens:
            new_tokens = agent_state.conversation_history_tokens[-1:]

        input_tokens = torch.tensor([new_tokens], device=self.accelerator.device)

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
            kv_cache=agent_state.main_cache,
            draft_cache=agent_state.draft_cache,
        )

        newly_generated_tokens = []
        for chunk, surprise in unwrapped_model.generate(gen_input):
            newly_generated_tokens.extend(chunk[0].tolist())
            if agent_state.complexity_manager:
                agent_state.complexity_manager.update_surprise(surprise)
        return newly_generated_tokens

    def _process_tool_call(self, agent_state: AgentState) -> bool:
        """Processes a tool call if one is present in the conversation history."""
        full_history_text = self.tokenizer.decode(agent_state.conversation_history_tokens)
        tool_name, args = parse_tool_call(full_history_text)

        if tool_name and args is not None:
            tool_output = self.tool_registry.execute_tool(tool_name, args)
            logging.info("Output of tool '%s':\n%s", tool_name, tool_output)

            tool_output_formatted = f"<TOOL_OUTPUT>{tool_output}</TOOL_OUTPUT>"
            agent_state.append_tokens(self.tokenizer.encode(tool_output_formatted))
            return True
        return False
