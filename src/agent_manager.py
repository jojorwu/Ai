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

from src.agent import Agent
from src.data_loader import get_batches_torch
from src.model import Transformer


@dataclass
class SpecializationConfig:
    """Configuration for the agent specialization process."""
    full_data: list
    seq_len: int
    batch_size: int
    steps_per_agent: int

@dataclass
class CollaborationContext:
    """Context for handling a collaboration request."""
    proposer: Agent
    prompt_tokens: torch.Tensor
    scores: dict
    proposer_index: int
    response: torch.Tensor

@dataclass
class IndependentResponseContext:
    """Context for handling an independent response."""
    proposer: Agent
    prompt_tokens: torch.Tensor
    scores: dict
    response: torch.Tensor

class AgentManager:
    """
    Manages the creation, specialization, and evaluation of a population of agents.
    """
    REWARD_INDEPENDENT_SUCCESS = 5.0
    PENALTY_INDEPENDENT_FAILURE = -5.0
    REWARD_GOOD_HELP = 3.0
    PENALTY_BAD_HELP = -3.0
    REWARD_ASKING_FOR_HELP = 0.5
    REWARD_ADMIT_IGNORANCE = 1.0
    SUCCESS_THRESHOLD = 0.5

    def __init__(
        self,
        base_model: Transformer,
        num_agents: int,
        accelerator: Accelerator,
    ):
        self.base_model = accelerator.unwrap_model(base_model)
        self.num_agents = num_agents
        self.agents: List[Agent] = []
        self.accelerator = accelerator
        self.fork_agents()

    def fork_agents(self):
        """Creates (clones) a population of agents from the base model."""
        logging.info("Cloning %d agents from the base model...", self.num_agents)
        for i in range(self.num_agents):
            agent = Agent(self.base_model, agent_id=f"agent_{i}")
            self.agents.append(agent)
        logging.info("Agents cloned successfully.")

    def specialize_agents_on_dataset(self, spec_config: SpecializationConfig, device):
        """Conducts a "specialization" phase."""
        if not spec_config.full_data:
            return
        data_tensor = torch.tensor(spec_config.full_data)
        data_chunks = torch.tensor_split(data_tensor, self.num_agents)
        logging.info("Specializing agents on different data subsets...")
        for i, agent in enumerate(self.agents):
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
                agent.experience(x_batch, y_batch)
                steps_done += 1
            if steps_done < spec_config.steps_per_agent:
                logging.warning(
                    "    - Only %d/%d steps were performed for %s.",
                    steps_done, spec_config.steps_per_agent, agent.agent_id
                )
        logging.info("Agent specialization complete.")

    def merge_agents(self, best_agents: List[Agent]):
        """Averages the LTM state_dicts of the 'best' agents."""
        if not best_agents:
            return
        ltm_states = [a.get_ltm_state() for a in best_agents if a.get_ltm_state()]
        if not ltm_states:
            logging.warning("None of the best agents had a valid LTM state.")
            return

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

    def _initialize_evaluation(self, tokenizer):
        """Initializes scores and token IDs for the evaluation phase."""
        scores = {agent.agent_id: 0.0 for agent in self.agents}
        tokens = {
            "ask_help": tokenizer.char_to_idx.get("<ASK_FOR_HELP>"),
            "i_dont_know": tokenizer.char_to_idx.get("<I_DONT_KNOW>"),
        }
        return scores, tokens

    def _handle_collaboration_request(self, ctx: CollaborationContext):
        """
        Handles the scenario where an agent asks for help.

        The proposing agent gets a small reward for asking. A helper agent is
        chosen to provide a response, and this response is then critiqued by all
        other agents. The helper and proposer are rewarded or penalized based on
        the critique score.
        """
        # Reward the agent for asking for help
        ctx.scores[ctx.proposer.agent_id] += self.REWARD_ASKING_FOR_HELP

        # Select the next agent in the list as the helper
        helper = self.agents[(ctx.proposer_index + 1) % len(self.agents)]
        if helper.agent_id == ctx.proposer.agent_id:
            return  # Avoid agent helping itself in a small population

        # Generate a response from the helper agent
        context = torch.cat([ctx.prompt_tokens, ctx.response], dim=1).to(
            helper.base_model.device
        )
        helper_response = helper.generate_response(context)
        new_helper_tokens = helper_response[:, context.shape[1] :]

        # All other agents act as critics
        critics = [
            a
            for a in self.agents
            if a.agent_id not in [ctx.proposer.agent_id, helper.agent_id]
        ] or [ctx.proposer]  # If no other critics, proposer critiques
        critic_ltms = [c.long_term_memory for c in critics]

        # Get the average critique score for the helper's response
        full_critique_sequence = torch.cat([context, new_helper_tokens], dim=1)
        avg_critique_score = self._batch_critique(
            full_critique_sequence.expand(len(critics), -1), critic_ltms
        )

        # Reward or penalize based on the critique
        if avg_critique_score > self.SUCCESS_THRESHOLD:
            reward = self.REWARD_GOOD_HELP * avg_critique_score
            ctx.scores[helper.agent_id] += reward
            ctx.scores[ctx.proposer.agent_id] += reward  # Proposer also rewarded for good question
        else:
            penalty = self.PENALTY_BAD_HELP * (1 - avg_critique_score)
            ctx.scores[helper.agent_id] += penalty

    def _handle_independent_response(self, ctx: IndependentResponseContext):
        """
        Handles the scenario where an agent responds independently.

        The response is critiqued by all other agents. The proposing agent is
        rewarded or penalized based on the average critique score.
        """
        # All other agents act as critics
        critics = (
            [a for a in self.agents if a.agent_id != ctx.proposer.agent_id]
            or [ctx.proposer]  # If no other critics, proposer critiques itself
        )
        new_response_tokens = ctx.response[:, ctx.prompt_tokens.shape[1] :]

        # Prepare for batch critique
        full_sequence = torch.cat([ctx.prompt_tokens, new_response_tokens], dim=1).expand(
            len(critics), -1
        )
        critic_ltms = [critic.long_term_memory for critic in critics]

        # Get the average critique score
        avg_critique_score = self._batch_critique(full_sequence, critic_ltms)

        # Reward or penalize the proposer based on the critique
        if avg_critique_score > self.SUCCESS_THRESHOLD:
            reward = self.REWARD_INDEPENDENT_SUCCESS * avg_critique_score
            ctx.scores[ctx.proposer.agent_id] += reward
        else:
            penalty = self.PENALTY_INDEPENDENT_FAILURE * (1 - avg_critique_score)
            ctx.scores[ctx.proposer.agent_id] += penalty

    def _batch_critique(self, full_sequence: torch.Tensor, critic_ltms: List[nn.Module]) -> float:
        """
        Performs a batched critique of a response using the base model.

        This method leverages the base model's value head to evaluate a sequence.
        It runs a single forward pass with a batch of LTM states from different
        critic agents, making the process highly efficient.

        Args:
            full_sequence: The complete token sequence (prompt + response) to be evaluated.
            critic_ltms: A list of LTM modules from the critic agents.

        Returns:
            The average value score from all critic agents.
        """
        with torch.no_grad():
            # Manually compute embeddings and LTM states to create a batch
            h = self.base_model.layers.embedding(full_sequence) * math.sqrt(
                self.base_model.config.model.d_model
            )

            ltm_states = torch.zeros(
                (len(critic_ltms), 1, h.size(2)), device=h.device, dtype=h.dtype
            )
            for i, ltm in enumerate(critic_ltms):
                if ltm:
                    # Each LTM processes its corresponding sequence embedding
                    ltm_input = h[i].mean(dim=0, keepdim=True).unsqueeze(0)
                    ltm_states[i] = ltm(ltm_input)

            # A single forward pass with the batched LTM states
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

        for i in range(min(num_prompts, self.num_agents * 2)):
            prompt_tokens_list = evaluation_data[i * prompt_len : (i + 1) * prompt_len]
            prompt_tensor = torch.tensor([prompt_tokens_list], device=device)
            self._evaluate_prompt_with_agents(prompt_tensor, scores, tokens)

    def _evaluate_prompt_with_agents(self, prompt_tensor, scores, tokens):
        """
        Orchestrates the evaluation of a single prompt by having each agent propose a
        response and then scoring that response based on the agent's behavior
        (e.g., responding independently, asking for help).
        """
        for i, proposer in enumerate(self.agents):
            response = proposer.generate_response(prompt_tensor)
            response_list = response[0].tolist()

            if tokens["ask_help"] in response_list:
                ctx = CollaborationContext(
                    proposer=proposer, prompt_tokens=prompt_tensor, scores=scores,
                    proposer_index=i, response=response
                )
                self._handle_collaboration_request(ctx)
            elif tokens["i_dont_know"] in response_list:
                scores[proposer.agent_id] += self.REWARD_ADMIT_IGNORANCE
            else:
                ctx = IndependentResponseContext(
                    proposer=proposer, prompt_tokens=prompt_tensor,
                    scores=scores, response=response
                )
                self._handle_independent_response(ctx)

    def collaborative_evaluation(self, evaluation_data, tokenizer, top_k, device):
        """
        Evaluates agents with a nuanced, peer-review-based scoring system.

        This method orchestrates a series of evaluations where agents propose
        responses to prompts. Agents can respond independently, ask for help,
        or admit ignorance. The quality of their actions is judged by their
        peers (other agents), leading to a fitness score.

        Args:
            evaluation_data: A list of token IDs for evaluation.
            tokenizer: The tokenizer instance.
            top_k: The number of top-performing agents to return.
            device: The device to run the evaluation on.

        Returns:
            A list of the top-k best performing agents.
        """
        if not self.agents or len(self.agents) < 2:
            logging.warning("Collaborative evaluation requires at least 2 agents.")
            return self.agents[:top_k]

        scores, tokens = self._initialize_evaluation(tokenizer)
        self._process_evaluation_prompts(evaluation_data, scores, tokens, device)
        return self._finalize_evaluation(scores, top_k)
