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
from src.agent.trainer import AgentTrainer
from src.data.data_loader import get_batches_torch
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
        """
        self.base_model = accelerator.unwrap_model(base_model)
        self.num_agents = num_agents
        self.agents: List[Agent] = []
        self.accelerator = accelerator
        self.evaluator = evaluator
        self.ltm_config = LTMConfig(
            learning_rate=self.base_model.config.ltm.optimizer.learning_rate,
            surprise_threshold=self.base_model.config.ltm.surprise_threshold,
        )
        self.fork_agents()

    def fork_agents(self):
        """
        Creates a population of agents by cloning the base model.
        Each agent shares the base model's weights but has a unique LTM.
        """
        logging.info("Cloning %d agents from the base model...", self.num_agents)
        for i in range(self.num_agents):
            agent = Agent(
                base_model=self.base_model,
                ltm_config=self.ltm_config,
                agent_id=f"agent_{i}",
            )
            self.agents.append(agent)
        logging.info("Agents cloned successfully.")

    def specialize_agents_on_dataset(self, spec_config: SpecializationConfig, device):
        """
        Conducts a "specialization" phase where each agent is trained on a
        unique subset of the data, managed by its own AgentTrainer.
        """
        if not spec_config.full_data:
            return
        data_tensor = torch.tensor(spec_config.full_data)
        data_chunks = torch.tensor_split(data_tensor, self.num_agents)
        logging.info("Specializing agents on different data subsets...")
        for i, agent in enumerate(self.agents):
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
                    steps_done, spec_config.steps_per_agent, agent.agent_id
                )
        logging.info("Agent specialization complete.")

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
