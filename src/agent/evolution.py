"""
This module contains the EvolutionaryOrchestrator class, which handles
population-level evolutionary logic like weight merging.
"""
import logging
import math
import torch
from typing import List

from src.agent.agent import Agent


class EvolutionaryOrchestrator:
    """
    Handles evolutionary operations for a population of agents.
    """

    @staticmethod
    def merge_population(best_agents: List[Agent], base_model: torch.nn.Module):
        """
        Merges the knowledge of the best-performing agents into the base model
        by averaging their LTM weights.
        """
        if not best_agents:
            return

        # Filter agents who haven't been successfully evaluated (fitness still -inf)
        best_agents = [a for a in best_agents if a.get_fitness_score() > -float('inf')]
        if not best_agents:
            logging.warning("No agents with valid fitness scores were provided for merging.")
            return

        ltm_states = [a.get_ltm_state() for a in best_agents if a.get_ltm_state()]
        if not ltm_states:
            logging.warning("None of the fit agents had a valid LTM state.")
            return

        # Determine target device for averaging (use base model's device)
        unwrapped_model = (
            base_model.module if hasattr(base_model, "module") else base_model
        )
        if not unwrapped_model.layers.long_term_memory:
            logging.warning("Base model does not have an LTM module to update.")
            return

        device = unwrapped_model.layers.long_term_memory.parameters().__next__().device
        logging.info(
            "Merging LTM weights from %d best agents on %s...", len(ltm_states), device
        )

        # Filter again to ensure we have valid fitness scores
        valid_agents = [a for a in best_agents if math.isfinite(a.get_fitness_score())]
        if not valid_agents:
            logging.warning("No agents with finite fitness scores found. Skipping merge.")
            return

        # Calculate weights based on fitness scores using Softmax to ensure they sum to 1.
        fitness_scores = torch.tensor(
            [a.get_fitness_score() for a in valid_agents],
            device=device,
            dtype=torch.float32,
        )

        # Handle cases where all agents have the same fitness (e.g., initial state)
        if len(valid_agents) == 1:
            weights = torch.tensor([1.0], device=device)
        elif torch.all(fitness_scores == fitness_scores[0]):
            weights = torch.full_like(fitness_scores, 1.0 / len(valid_agents))
        else:
            weights = torch.softmax(fitness_scores, dim=0)

        # Re-assign best_agents and ltm_states to match valid_agents
        best_agents = valid_agents
        ltm_states = [a.get_ltm_state() for a in best_agents if a.get_ltm_state()]

        logging.info(
            "Merging LTM weights from %d agents using weighted averaging (max weight: %.4f)...",
            len(best_agents),
            weights.max().item(),
        )

        # Initialize a dictionary for the weighted averaged state with zero-tensors.
        avg_state = {
            key: torch.zeros_like(tensor, device=device)
            for key, tensor in ltm_states[0].items()
        }

        # Perform weighted averaging on-device.
        for i, state in enumerate(ltm_states):
            weight = weights[i]
            for key, tensor in state.items():
                avg_state[key].add_(tensor.to(device) * weight)

        # Update the base model's LTM, ensuring all merged weights are finite.
        if all(torch.isfinite(t).all() for t in avg_state.values()):
            unwrapped_model.layers.long_term_memory.load_state_dict(avg_state)
            logging.info("Base model's LTM has been updated with merged weights.")
        else:
            logging.error(
                "Merged LTM state contains non-finite values (NaN/Inf). "
                "Skipping update to base model to prevent corruption."
            )
