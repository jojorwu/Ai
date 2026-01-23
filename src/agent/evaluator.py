"""
Handles the collaborative evaluation of agents.
"""
import logging
import math
from typing import List

import torch
from torch import nn

from src.agent.agent import Agent
from src.agent.dataclasses import (
    CollaborationContext,
    CritiqueScoringContext,
    IndependentResponseContext,
)


class CollaborativeEvaluator:
    """
    Manages the collaborative evaluation of a population of agents.
    """

    REWARD_INDEPENDENT_SUCCESS = 5.0
    PENALTY_INDEPENDENT_FAILURE = -5.0
    REWARD_GOOD_HELP = 3.0
    PENALTY_BAD_HELP = -3.0
    REWARD_ASKING_FOR_HELP = 0.5
    REWARD_ADMIT_IGNORANCE = 1.0
    SUCCESS_THRESHOLD = 0.5

    def __init__(self, agents: List[Agent], base_model: nn.Module):
        self.agents = agents
        self.base_model = base_model

    def collaborative_evaluation(self, evaluation_data, tokenizer, top_k, device):
        """
        Evaluates agents with a nuanced, peer-review-based scoring system.
        """
        if not self.agents or len(self.agents) < 2:
            logging.warning("Collaborative evaluation requires at least 2 agents.")
            return self.agents[:top_k]

        scores, tokens = self._initialize_evaluation(tokenizer)
        self._process_evaluation_prompts(evaluation_data, scores, tokens, device)
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

    def _get_critics(self, proposer: Agent, agents_to_exclude: List[Agent]) -> List[Agent]:
        """
        Selects agents from the population to act as critics.
        """
        exclude_ids = {agent.agent_id for agent in agents_to_exclude}
        critics = [
            agent for agent in self.agents if agent.agent_id not in exclude_ids
        ]
        return critics or [proposer]

    def _apply_critique_based_scores(self, csc: CritiqueScoringContext):
        """
        Applies rewards or penalties to agents based on a critique score.
        """
        if csc.avg_critique_score > self.SUCCESS_THRESHOLD:
            reward = csc.success_reward * csc.avg_critique_score
            for agent in csc.agents_to_reward:
                csc.scores[agent.agent_id] += reward
        else:
            penalty = csc.failure_penalty * (1 - csc.avg_critique_score)
            for agent in csc.agents_to_penalize:
                csc.scores[agent.agent_id] += penalty

    def _handle_collaboration_request(self, ctx: CollaborationContext):
        """
        Manages the 'ask for help' scenario in collaborative evaluation.
        """
        ctx.scores[ctx.proposer.agent_id] += self.REWARD_ASKING_FOR_HELP

        helper = self.agents[(ctx.proposer_index + 1) % len(self.agents)]
        if helper.agent_id == ctx.proposer.agent_id:
            return

        context = torch.cat([ctx.prompt_tokens, ctx.response], dim=1).to(
            helper.base_model.device
        )
        helper_response = helper.generate_response(context)
        new_helper_tokens = helper_response[:, context.shape[1] :]

        critics = self._get_critics(ctx.proposer, [ctx.proposer, helper])
        critic_ltms = [c.long_term_memory for c in critics]

        full_critique_sequence = torch.cat([context, new_helper_tokens], dim=1)
        avg_critique_score = self._batch_critique(
            full_critique_sequence.expand(len(critics), -1), critic_ltms
        )
        csc = CritiqueScoringContext(
            scores=ctx.scores,
            avg_critique_score=avg_critique_score,
            success_reward=self.REWARD_GOOD_HELP,
            failure_penalty=self.PENALTY_BAD_HELP,
            agents_to_reward=[helper, ctx.proposer],
            agents_to_penalize=[helper],
        )
        self._apply_critique_based_scores(csc)

    def _handle_independent_response(self, ctx: IndependentResponseContext):
        """
        Manages the 'independent response' scenario in collaborative evaluation.
        """
        critics = self._get_critics(ctx.proposer, [ctx.proposer])
        new_response_tokens = ctx.response[:, ctx.prompt_tokens.shape[1] :]

        full_sequence = torch.cat(
            [ctx.prompt_tokens, new_response_tokens], dim=1
        ).expand(len(critics), -1)
        critic_ltms = [critic.long_term_memory for critic in critics]

        avg_critique_score = self._batch_critique(full_sequence, critic_ltms)

        csc = CritiqueScoringContext(
            scores=ctx.scores,
            avg_critique_score=avg_critique_score,
            success_reward=self.REWARD_INDEPENDENT_SUCCESS,
            failure_penalty=self.PENALTY_INDEPENDENT_FAILURE,
            agents_to_reward=[ctx.proposer],
            agents_to_penalize=[ctx.proposer],
        )
        self._apply_critique_based_scores(csc)

    def _batch_critique(self, full_sequence: torch.Tensor, critic_ltms: List[nn.Module]) -> float:
        """
        Performs a batched critique of a response using the base model.
        """
        with torch.no_grad():
            h = self.base_model.layers.embedding(full_sequence) * math.sqrt(
                self.base_model.config.model.d_model
            )
            ltm_states = torch.zeros(
                (len(critic_ltms), 1, h.size(2)), device=h.device, dtype=h.dtype
            )
            for i, ltm in enumerate(critic_ltms):
                if ltm:
                    ltm_input = h[i].mean(dim=0, keepdim=True).unsqueeze(0)
                    ltm_states[i], _ = ltm(ltm_input)
            _, values, _ = self.base_model.forward(
                full_sequence, ltm_state=ltm_states
            )
        return values.mean().item()

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
            logging.info(
                "    - %s: %.4f", agent.agent_id, agent.get_fitness_score()
            )
        return sorted_agents[:top_k]

    def _process_evaluation_prompts(self, evaluation_data, scores, tokens, device):
        """
        Iterates through evaluation data, creating prompts and triggering evaluation for each.
        """
        prompt_len = self.base_model.config.model.max_seq_len // 2
        num_prompts = len(evaluation_data) // prompt_len
        if num_prompts == 0:
            logging.warning("Not enough validation data for evaluation.")
            return

        for i in range(min(num_prompts, len(self.agents) * 2)):
            prompt_tokens_list = evaluation_data[i * prompt_len : (i + 1) * prompt_len]
            prompt_tensor = torch.tensor([prompt_tokens_list], device=device)
            self._evaluate_prompt_with_agents(prompt_tensor, scores, tokens)

    def _evaluate_prompt_with_agents(self, prompt_tensor, scores, tokens):
        """
        Orchestrates the evaluation of a single prompt by having each agent propose a response.
        """
        for i, proposer in enumerate(self.agents):
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
                self._handle_collaboration_request(ctx)
            elif tokens["i_dont_know"] in response_list:
                scores[proposer.agent_id] += self.REWARD_ADMIT_IGNORANCE
            else:
                ctx = IndependentResponseContext(
                    proposer=proposer,
                    prompt_tokens=prompt_tensor,
                    scores=scores,
                    response=response,
                )
                self._handle_independent_response(ctx)
