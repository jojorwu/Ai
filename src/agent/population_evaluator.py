"""
This module contains the PopulationEvaluator class, which manages the
outer loop of collaborative evaluation.
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, TYPE_CHECKING

import torch

from src.agent.agent import Agent
from src.agent.dataclasses import CollaborationContext, IndependentResponseContext

if TYPE_CHECKING:
    from src.agent.evaluator import CollaborativeEvaluator


class PopulationEvaluator:
    """
    Orchestrates the evaluation of a population of agents by iterating
    through evaluation data and coordinating with a CollaborativeEvaluator.
    """

    def process_evaluation_data(
        self,
        evaluation_data,
        agents: List[Agent],
        scores: dict,
        tokens: dict,
        device: torch.device,
        max_seq_len: int,
        collaborative_evaluator: "CollaborativeEvaluator",
    ):
        """
        Iterates through evaluation data, creating prompts and triggering
        evaluation for each agent.
        """
        prompt_len = max_seq_len // 2
        num_prompts = len(evaluation_data) // prompt_len
        if num_prompts == 0:
            logging.warning("Not enough validation data for evaluation.")
            return

        # Cap the number of evaluation rounds to keep it efficient
        num_rounds = min(num_prompts, len(agents) * 2)

        for i in range(num_rounds):
            # Optimized prompt creation: avoid redundant slicing and list creation
            prompt_tensor = (
                torch.as_tensor(evaluation_data[i * prompt_len : (i + 1) * prompt_len])
                .unsqueeze(0)
                .to(device)
            )

            self._evaluate_prompt(
                prompt_tensor, agents, scores, tokens, collaborative_evaluator
            )

    def _evaluate_prompt(
        self,
        prompt_tensor: torch.Tensor,
        agents: List[Agent],
        scores: dict,
        tokens: dict,
        collaborative_evaluator: "CollaborativeEvaluator",
    ):
        """
        Orchestrates the evaluation of a single prompt by having each agent
        propose a response and delegating the scoring to CollaborativeEvaluator.
        """

        def _get_response_and_score(i, proposer):
            try:
                response = proposer.generate_response(prompt_tensor)

                # Vectorized token check to avoid expensive tolist() and GPU-CPU sync.
                # Gracefully handle cases where special tokens might be missing (None).
                has_ask_help = False
                if tokens["ask_help"] is not None:
                    has_ask_help = (response == tokens["ask_help"]).any().item()

                has_i_dont_know = False
                if tokens["i_dont_know"] is not None:
                    has_i_dont_know = (response == tokens["i_dont_know"]).any().item()

                if has_ask_help:
                    ctx = CollaborationContext(
                        proposer=proposer,
                        prompt_tokens=prompt_tensor,
                        scores=scores,
                        proposer_index=i,
                        response=response,
                    )
                    collaborative_evaluator.handle_collaboration_request(ctx)
                elif has_i_dont_know:
                    # Direct scoring for simple responses
                    scoring_engine = collaborative_evaluator.scoring_engine
                    scores[proposer.agent_id] += scoring_engine.reward_admit_ignorance
                else:
                    ctx = IndependentResponseContext(
                        proposer=proposer,
                        prompt_tokens=prompt_tensor,
                        scores=scores,
                        response=response,
                    )
                    collaborative_evaluator.handle_independent_response(ctx)
            except Exception as e:  # pylint: disable=broad-except
                logging.error("Failed to evaluate agent %s: %s", proposer.agent_id, e)

        max_workers = min(len(agents), os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_get_response_and_score, i, proposer)
                for i, proposer in enumerate(agents)
            ]
            for future in futures:
                future.result()
