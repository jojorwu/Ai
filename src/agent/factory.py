"""
Factory for creating Agent instances.
"""
import logging
from typing import List

from src.agent.agent import Agent
from src.agent.dataclasses import LTMConfig
from src.model.model import Transformer


class AgentFactory:
    """
    Handles the creation and population of agents.
    """

    @staticmethod
    def create_population(
        base_model: Transformer,
        num_agents: int,
        ltm_config: LTMConfig
    ) -> List[Agent]:
        """
        Creates a population of agents by cloning the base model.
        Each agent shares the base model's weights but has a unique LTM.
        """
        logging.info("Cloning %d agents from the base model...", num_agents)
        agents = []
        for i in range(num_agents):
            agent = Agent(
                base_model=base_model,
                ltm_config=ltm_config,
                agent_id=f"agent_{i}",
            )
            agents.append(agent)
        logging.info("Agents cloned successfully.")
        return agents
