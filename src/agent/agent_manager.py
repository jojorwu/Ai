"""
PyTorch implementation of the AgentManager for managing the agent lifecycle.
"""
import logging
import math
from dataclasses import dataclass
from typing import List

import torch
from accelerate import Accelerator
from torch import nn

from src.agent.agent import Agent
from src.agent.dataclasses import LTMConfig, SpecializationConfig
from src.agent.evaluator import CollaborativeEvaluator
from src.agent.factory import AgentFactory
from src.agent.specializer import AgentSpecializer
from src.model.model import Transformer


class AgentManager:
    """
    Manages the creation, specialization, and evaluation of a population of agents.
    """

    def __init__(
        self,
        base_model: Transformer,
        num_agents: int,
        accelerator: Accelerator,
        evaluator: CollaborativeEvaluator,
    ):
        """
        Initializes the AgentManager.

        Args:
            base_model: The base Transformer model to be shared among agents.
            num_agents: The number of agents to create in the population.
            accelerator: The Accelerator object for distributed training.
            evaluator: The evaluator used for agent performance assessment.
        """
        self.base_model = accelerator.unwrap_model(base_model)
        self.num_agents = num_agents
        self.accelerator = accelerator
        self.evaluator = evaluator

        self.ltm_config = LTMConfig(
            learning_rate=self.base_model.config.ltm.optimizer.learning_rate,
            surprise_threshold=self.base_model.config.ltm.surprise_threshold,
        )

        # Use AgentFactory for agent creation
        self.agents = AgentFactory.create_population(
            self.base_model, self.num_agents, self.ltm_config
        )
        self.specializer = AgentSpecializer()

    def specialize_agents_on_dataset(self, spec_config: SpecializationConfig, device):
        """
        Delegates the specialization phase to the AgentSpecializer.
        """
        self.specializer.specialize_agents(self.agents, spec_config, device)

    def merge_agents(self, best_agents: List[Agent]):
        """
        Merges the knowledge of the best-performing agents into the base model.

        This is achieved by averaging the weights (state_dict) of the Long-Term
        Memory (LTM) modules from the provided list of 'best' agents. The resulting
        averaged LTM state is then loaded into the base model's LTM, effectively
        assimilating the collective knowledge of the top performers.

        Args:
            best_agents: A list of the top-performing Agent objects from which
                         to merge LTM states.
        """
        if not best_agents:
            return
        ltm_states = [a.get_ltm_state() for a in best_agents if a.get_ltm_state()]
        if not ltm_states:
            logging.warning("None of the best agents had a valid LTM state.")
            return

        # Initialize a dictionary for the averaged state with zero-tensors.
        # This avoids modifying any of the original state_dicts.
        avg_state = {
            key: torch.zeros_like(tensor, device="cpu")
            for key, tensor in ltm_states[0].items()
        }
        for state in ltm_states:
            for key, tensor in state.items():
                avg_state[key] += tensor.to("cpu")
        for key in avg_state:
            avg_state[key] /= len(ltm_states)

        model_to_update = (
            self.base_model.module
            if hasattr(self.base_model, "module")
            else self.base_model
        )
        if model_to_update.layers.long_term_memory:
            model_to_update.layers.long_term_memory.load_state_dict(avg_state)
            logging.info("Base model's LTM has been updated with merged weights.")

    def collaborative_evaluation(self, evaluation_data, tokenizer, top_k, device):
        """
        Delegates the collaborative evaluation to the CollaborativeEvaluator.
        """
        return self.evaluator.collaborative_evaluation(
            evaluation_data, tokenizer, top_k, device
        )
