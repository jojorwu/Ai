"""
This module contains the AgentSpecializer class, which is responsible for
training agents on unique subsets of data.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List

import torch

from src.agent.agent import Agent
from src.agent.dataclasses import SpecializationConfig
from src.agent.trainer import AgentTrainer
from src.data.data_loader import get_batches_torch


class AgentSpecializer:
    """
    Handles the specialization phase where each agent is trained on a
    unique subset of the data.
    """

    def specialize_agents(
        self,
        agents: List[Agent],
        spec_config: SpecializationConfig,
        device: torch.device,
    ):
        """
        Conducts a "specialization" phase where each agent is trained on a
        unique subset of the data, managed by its own AgentTrainer.
        """
        if not spec_config.full_data:
            logging.warning("No data provided for specialization.")
            return

        num_agents = len(agents)
        if num_agents == 0:
            logging.warning("No agents provided for specialization.")
            return

        data_tensor = torch.tensor(spec_config.full_data)
        data_chunks = torch.tensor_split(data_tensor, num_agents)

        logging.info("Specializing %d agents in parallel...", num_agents)

        def _train_single_agent(idx, agent):
            agent_trainer = AgentTrainer(agent)
            agent_data = data_chunks[idx].tolist()

            if not agent_data or len(agent_data) < spec_config.seq_len + 1:
                logging.info("  - Skipping %s, not enough data.", agent.agent_id)
                return

            batch_generator = get_batches_torch(
                agent_data, spec_config.batch_size, spec_config.seq_len, device
            )

            steps_done = 0
            for x_batch, y_batch, _ in batch_generator:
                if steps_done >= spec_config.steps_per_agent:
                    break
                agent_trainer.experience(x_batch, y_batch)
                steps_done += 1

            logging.info("  - Specialized %s in %d steps.", agent.agent_id, steps_done)

        with ThreadPoolExecutor(max_workers=num_agents) as executor:
            for i, agent in enumerate(agents):
                executor.submit(_train_single_agent, i, agent)

        logging.info("Agent specialization complete.")

    def _specialize_agents_sequential(
        self,
        agents: List[Agent],
        spec_config: SpecializationConfig,
        device: torch.device,
    ):
        """
        Conducts a "specialization" phase where each agent is trained on a
        unique subset of the data, managed by its own AgentTrainer.
        """
        if not spec_config.full_data:
            logging.warning("No data provided for specialization.")
            return

        data_tensor = torch.tensor(spec_config.full_data)
        num_agents = len(agents)
        if num_agents == 0:
            logging.warning("No agents provided for specialization.")
            return

        data_chunks = torch.tensor_split(data_tensor, num_agents)

        logging.info("Specializing %d agents on different data subsets...", num_agents)

        for i, agent in enumerate(agents):
            agent_trainer = AgentTrainer(agent)
            agent_data = data_chunks[i].tolist()

            if not agent_data or len(agent_data) < spec_config.seq_len + 1:
                logging.info("  - Skipping %s, not enough data.", agent.agent_id)
                continue

            logging.info(
                "  - Specializing %s on %d items...", agent.agent_id, len(agent_data)
            )

            batch_generator = get_batches_torch(
                agent_data, spec_config.batch_size, spec_config.seq_len, device
            )

            steps_done = 0
            for x_batch, y_batch, _ in batch_generator:
                if steps_done >= spec_config.steps_per_agent:
                    break
                agent_trainer.experience(x_batch, y_batch)
                steps_done += 1

            if steps_done < spec_config.steps_per_agent:
                logging.warning(
                    "    - Only %d/%d steps were performed for %s.",
                    steps_done,
                    spec_config.steps_per_agent,
                    agent.agent_id,
                )

        logging.info("Agent specialization complete.")
