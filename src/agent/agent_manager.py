"""
PyTorch implementation of the AgentManager for managing the agent lifecycle.
"""
import logging
import math
from dataclasses import dataclass
from typing import Any, List

import torch
from accelerate import Accelerator
from torch import nn

from src.agent.agent import Agent
from src.agent.dataclasses import LTMConfig, SpecializationConfig
from src.agent.evaluator import CollaborativeEvaluator
from src.agent.evolution import EvolutionaryOrchestrator
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
        hardware_config=None,
    ):
        """
        Initializes the AgentManager.

        Args:
            base_model: The base Transformer model to be shared among agents.
            num_agents: The number of agents to create in the population.
            accelerator: The Accelerator object for distributed training.
            evaluator: The evaluator used for agent performance assessment.
            hardware_config: Hardware configuration settings.
        """
        self.base_model = accelerator.unwrap_model(base_model)
        self.num_agents = num_agents
        self.accelerator = accelerator
        self.evaluator = evaluator
        self.hardware_config = hardware_config

        self.ltm_config = LTMConfig(
            learning_rate=self.base_model.config.ltm.optimizer.learning_rate,
            surprise_threshold=self.base_model.config.ltm.surprise_threshold,
        )

        # Use AgentFactory for agent creation
        self.agents = AgentFactory.create_population(
            self.base_model, self.num_agents, self.ltm_config
        )
        self.specializer = AgentSpecializer()

    def specialize_agents_on_dataset(
        self, spec_config: SpecializationConfig, device: torch.device
    ) -> None:
        """
        Delegates the specialization phase to the AgentSpecializer.
        """
        self.specializer.specialize_agents(
            self.agents, spec_config, device, self.hardware_config
        )

    def merge_agents(self, best_agents: List[Agent]) -> None:
        """
        Delegates the merging of agent knowledge to the EvolutionaryOrchestrator.
        """
        EvolutionaryOrchestrator.merge_population(best_agents, self.base_model)

    def collaborative_evaluation(
        self, evaluation_data: list, tokenizer: Any, top_k: int, device: torch.device
    ) -> dict:
        """
        Delegates the collaborative evaluation to the CollaborativeEvaluator.
        """
        return self.evaluator.collaborative_evaluation(
            evaluation_data, tokenizer, top_k, device
        )
