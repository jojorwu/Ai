"""
PyTorch implementation of the AgentManager for managing the agent lifecycle.
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, List

import torch

from src.agent.dataclasses import LTMConfig, SpecializationConfig
from src.agent.specializer import AgentSpecializer
from src.agent.factory import AgentFactory
from src.agent.evolution import EvolutionaryOrchestrator

if TYPE_CHECKING:
    from accelerate import Accelerator
    from src.agent.agent import Agent
    from src.agent.evaluator import CollaborativeEvaluator
    from src.model.model import Transformer
    from src.config.hardware_config import HardwareConfig


class AgentManager:
    """
    Manages the creation, specialization, and evaluation of a population of agents.

    This class orchestrates the lifecycle of agents, from initial creation to
    specialization on datasets, collaborative evaluation, and final merging
    of the best agents' knowledge.
    """

    def __init__(
        self,
        base_model: Transformer,
        num_agents: int,
        accelerator: Accelerator,
        evaluator: CollaborativeEvaluator,
        hardware_config: HardwareConfig | None = None,
    ) -> None:
        """
        Initializes the AgentManager.

        Args:
            base_model: The base Transformer model shared among agents.
            num_agents: The number of agents to maintain in the population.
            accelerator: Accelerator instance for distributed training.
            evaluator: Evaluator for performance assessment.
            hardware_config: Optional hardware configuration settings.
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

        # Create the initial population of agents
        self.agents = AgentFactory.create_population(
            self.base_model, self.num_agents, self.ltm_config
        )
        self.specializer = AgentSpecializer()

    def specialize_agents_on_dataset(
        self, spec_config: SpecializationConfig, device: torch.device
    ) -> None:
        """
        Specializes the agent population on a specific dataset.

        Args:
            spec_config: Configuration for the specialization process.
            device: The device to run specialization on.
        """
        self.specializer.specialize_agents(
            self.agents, spec_config, device, self.hardware_config
        )

    def merge_agents(self, best_agents: List[Agent]) -> None:
        """
        Merges knowledge from the best performing agents back into the base model.

        Args:
            best_agents: A list of agents selected for merging.
        """
        EvolutionaryOrchestrator.merge_population(best_agents, self.base_model)

    def collaborative_evaluation(
        self,
        evaluation_data: list[Any],
        tokenizer: Any,
        top_k: int,
        device: torch.device,
    ) -> List[Agent]:
        """
        Performs a collaborative evaluation of the agents.

        Args:
            evaluation_data: Data to use for evaluation.
            tokenizer: Tokenizer for decoding responses.
            top_k: Number of top-performing agents to select.
            device: Device to run evaluation on.

        Returns:
            A list of the top-k best performing agents.
        """
        # The underlying evaluator returns a sorted list of Agent objects.
        return self.evaluator.collaborative_evaluation(
            evaluation_data, tokenizer, top_k, device
        )
