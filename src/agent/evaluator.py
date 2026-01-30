"""
Handles the collaborative evaluation of agents.
"""
import logging
import threading
from typing import List

import torch
from torch import nn

from src.agent.agent import Agent
from src.agent.dataclasses import (
    CollaborationContext,
    CritiqueScoringContext,
    IndependentResponseContext,
)
from src.agent.population_evaluator import PopulationEvaluator
from src.agent.scoring import ScoringEngine


class CollaborativeEvaluator:
    """
    Manages the collaborative evaluation of a population of agents.
    """

    def __init__(
        self,
        agents: List[Agent],
        base_model: nn.Module,
        scoring_engine: ScoringEngine = None,
    ):
        self.agents = agents
        self.base_model = base_model
        self.scoring_engine = scoring_engine or ScoringEngine()
        self.population_evaluator = PopulationEvaluator()
        self._score_lock = threading.Lock()

    def collaborative_evaluation(self, evaluation_data, tokenizer, top_k, device):
        """
        Evaluates agents with a nuanced, peer-review-based scoring system.
        """
        if not self.agents or len(self.agents) < 2:
            logging.warning("Collaborative evaluation requires at least 2 agents.")
            return self.agents[:top_k]

        scores, tokens = self._initialize_evaluation(tokenizer)

        self.population_evaluator.process_evaluation_data(
            evaluation_data=evaluation_data,
            agents=self.agents,
            scores=scores,
            tokens=tokens,
            device=device,
            max_seq_len=self.base_model.config.model.max_seq_len,
            collaborative_evaluator=self,
        )

        return self._finalize_evaluation(scores, top_k)

    def _initialize_evaluation(self, tokenizer):
        """
        Initializes the data structures needed for a collaborative evaluation cycle.
        """
        scores = {agent.agent_id: 0.0 for agent in self.agents}
        tokens = {
            "ask_help": tokenizer.char_to_idx.get("<ASK_FOR_HELP>"),
            "i_dont_know": tokenizer.char_to_idx.get("<I_DONT_KNOW>"),
        }
        return scores, tokens

    def _get_critics(
        self, proposer: Agent, agents_to_exclude: List[Agent]
    ) -> List[Agent]:
        """
        Selects top-performing agents from the population to act as critics.
        Higher fitness agents provide more reliable critiques.
        """
        exclude_ids = {agent.agent_id for agent in agents_to_exclude}

        # Sort agents by fitness (descending) and exclude proposer/helpers
        potential_critics = sorted(
            [a for a in self.agents if a.agent_id not in exclude_ids],
            key=lambda a: a.get_fitness_score(),
            reverse=True
        )

        # Use top 50% of available agents as critics (at least 1)
        num_critics = max(1, len(potential_critics) // 2)
        critics = potential_critics[:num_critics]

        return critics or [proposer]

    def handle_collaboration_request(self, ctx: CollaborationContext):
        """
        Manages the 'ask for help' scenario in collaborative evaluation.
        """
        with self._score_lock:
            ctx.scores[ctx.proposer.agent_id] += self.scoring_engine.reward_asking_for_help

        helper = self.agents[(ctx.proposer_index + 1) % len(self.agents)]
        if helper.agent_id == ctx.proposer.agent_id:
            return

        context = ctx.response.to(helper.base_model.device)
        helper_response = helper.generate_response(context)
        new_helper_tokens = helper_response[:, context.shape[1] :]

        critics = self._get_critics(ctx.proposer, [ctx.proposer, helper])
        critic_ltms = [c.long_term_memory for c in critics]

        full_critique_sequence = torch.cat([context, new_helper_tokens], dim=1)
        avg_critique_score = self.scoring_engine.calculate_critique_score(
            self.base_model,
            full_critique_sequence.expand(len(critics), -1),
            critic_ltms,
        )
        csc = CritiqueScoringContext(
            scores=ctx.scores,
            avg_critique_score=avg_critique_score,
            success_reward=self.scoring_engine.reward_good_help,
            failure_penalty=self.scoring_engine.penalty_bad_help,
            agents_to_reward=[helper, ctx.proposer],
            agents_to_penalize=[helper],
        )
        with self._score_lock:
            self.scoring_engine.apply_scores(csc)

    def handle_independent_response(self, ctx: IndependentResponseContext):
        """
        Manages the 'independent response' scenario in collaborative evaluation.
        """
        critics = self._get_critics(ctx.proposer, [ctx.proposer])
        full_sequence = ctx.response.expand(len(critics), -1)
        critic_ltms = [critic.long_term_memory for critic in critics]

        avg_critique_score = self.scoring_engine.calculate_critique_score(
            self.base_model, full_sequence, critic_ltms
        )

        csc = CritiqueScoringContext(
            scores=ctx.scores,
            avg_critique_score=avg_critique_score,
            success_reward=self.scoring_engine.reward_independent_success,
            failure_penalty=self.scoring_engine.penalty_independent_failure,
            agents_to_reward=[ctx.proposer],
            agents_to_penalize=[ctx.proposer],
        )
        with self._score_lock:
            self.scoring_engine.apply_scores(csc)

    def _finalize_evaluation(self, scores, top_k):
        """
        Updates agent fitness scores, logs the results, and returns the top-k agents.
        """
        for agent in self.agents:
            agent.update_fitness_score(scores[agent.agent_id])
        sorted_agents = sorted(
            self.agents, key=lambda a: a.get_fitness_score(), reverse=True
        )
        logging.info("  - Collaborative Evaluation Fitness scores:")
        for agent in sorted_agents:
            logging.info("    - %s: %.4f", agent.agent_id, agent.get_fitness_score())
        return sorted_agents[:top_k]
