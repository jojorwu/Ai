"""
This module contains the PopulationEvaluator class, which manages the
outer loop of collaborative evaluation.
"""
import logging
import torch
from typing import List, TYPE_CHECKING

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
            prompt_tokens_list = evaluation_data[i * prompt_len : (i + 1) * prompt_len]
            prompt_tensor = torch.tensor([prompt_tokens_list], device=device)

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
        for i, proposer in enumerate(agents):
            response = proposer.generate_response(prompt_tensor)
            response_list = response[0].tolist()

            if tokens["ask_help"] in response_list:
                ctx = CollaborationContext(
                    proposer=proposer,
                    prompt_tokens=prompt_tensor,
                    scores=scores,
                    proposer_index=i,
                    response=response,
                )
                collaborative_evaluator.handle_collaboration_request(ctx)
            elif tokens["i_dont_know"] in response_list:
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
