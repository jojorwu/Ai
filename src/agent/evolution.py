"""
This module contains the EvolutionaryOrchestrator class, which handles
population-level evolutionary logic like weight merging.
"""
import logging
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

        ltm_states = [a.get_ltm_state() for a in best_agents if a.get_ltm_state()]
        if not ltm_states:
            logging.warning("None of the best agents had a valid LTM state.")
            return

        logging.info("Merging LTM weights from %d best agents...", len(ltm_states))

        # Initialize a dictionary for the averaged state with zero-tensors.
        avg_state = {
            key: torch.zeros_like(tensor, device="cpu")
            for key, tensor in ltm_states[0].items()
        }
        for state in ltm_states:
            for key, tensor in state.items():
                avg_state[key] += tensor.to("cpu")

        for key in avg_state:
            avg_state[key] /= len(ltm_states)

        # Update the base model's LTM
        unwrapped_model = (
            base_model.module if hasattr(base_model, "module") else base_model
        )

        if unwrapped_model.layers.long_term_memory:
            unwrapped_model.layers.long_term_memory.load_state_dict(avg_state)
            logging.info("Base model's LTM has been updated with merged weights.")
        else:
            logging.warning("Base model does not have an LTM module to update.")
