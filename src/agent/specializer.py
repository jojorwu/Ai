"""
This module contains the AgentSpecializer class, which is responsible for
training agents on unique subsets of data.
"""
import logging
import os
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

        # Limit workers to avoid excessive thread overhead
        max_workers = min(num_agents, os.cpu_count() or 4)
        logging.info("Specializing %d agents in parallel using %d workers...", num_agents, max_workers)

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

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, agent in enumerate(agents):
                executor.submit(_train_single_agent, i, agent)

        logging.info("Agent specialization complete.")
