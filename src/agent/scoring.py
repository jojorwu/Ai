"""
Handles the scoring logic for collaborative evaluation.
"""
import logging
import math
from typing import List

import torch
from torch import nn

from src.agent.agent import Agent
from src.agent.dataclasses import CritiqueScoringContext


class ScoringEngine:
    """
    Encapsulates the nuanced scoring logic for peer-review-based evaluation.
    """

    def __init__(
        self,
        success_threshold: float = 0.5,
        reward_independent_success: float = 5.0,
        penalty_independent_failure: float = -5.0,
        reward_good_help: float = 3.0,
        penalty_bad_help: float = -3.0,
        reward_asking_for_help: float = 0.5,
        reward_admit_ignorance: float = 1.0,
    ):
        self.success_threshold = success_threshold
        self.reward_independent_success = reward_independent_success
        self.penalty_independent_failure = penalty_independent_failure
        self.reward_good_help = reward_good_help
        self.penalty_bad_help = penalty_bad_help
        self.reward_asking_for_help = reward_asking_for_help
        self.reward_admit_ignorance = reward_admit_ignorance

    def calculate_critique_score(
        self, base_model: nn.Module, full_sequence: torch.Tensor, critic_ltms: List[nn.Module]
    ) -> float:
        """
        Performs a batched critique of a response using the base model.
        """
        with torch.inference_mode():
            h = base_model.layers.embedding(full_sequence) * math.sqrt(
                base_model.config.model.d_model
            )
            ltm_states = torch.zeros(
                (len(critic_ltms), 1, h.size(2)), device=h.device, dtype=h.dtype
            )
            # Vectorized summary calculation across all critics
            summaries = h.mean(dim=1, keepdim=True)
            for i, ltm in enumerate(critic_ltms):
                if ltm:
                    ltm_states[i], _ = ltm(summaries[i : i + 1])
            _, values, _ = base_model.forward(
                full_sequence, ltm_state=ltm_states
            )
        return values.mean().item()

    def apply_scores(self, csc: CritiqueScoringContext):
        """
        Applies rewards or penalties to agents based on a critique score.
        """
        if csc.avg_critique_score > self.success_threshold:
            reward = csc.success_reward * csc.avg_critique_score
            for agent in csc.agents_to_reward:
                csc.scores[agent.agent_id] += reward
        else:
            penalty = csc.failure_penalty * (1 - csc.avg_critique_score)
            for agent in csc.agents_to_penalize:
                csc.scores[agent.agent_id] += penalty
