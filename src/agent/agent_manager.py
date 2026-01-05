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
from src.config import LTMConfig
from src.data.data_loader import get_batches_torch
from src.model.model import Transformer


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
        ltm_config: LTMConfig,
        accelerator: Accelerator,
    ):
        self.base_model = accelerator.unwrap_model(base_model)
        self.num_agents = num_agents
        self.ltm_config = ltm_config
        self.agents: List[Agent] = []
        self.accelerator = accelerator
        self.fork_agents()

    def fork_agents(self):
        """Creates (clones) a population of agents from the base model."""
        logging.info("Cloning %d agents from the base model...", self.num_agents)
        for i in range(self.num_agents):
            agent = Agent(self.base_model, self.ltm_config, agent_id=f"agent_{i}")
            self.agents.append(agent)
        logging.info("Agents cloned successfully.")

    def specialize_agents_on_dataset(self, spec_config: SpecializationConfig, device):
        """
        Trains each agent on a unique subset of the provided data.

        This "specialization" phase is crucial for fostering diversity. By exposing
        each agent to different data, their individual Long-Term Memory (LTM)
        modules adapt in unique ways, leading to a population with varied
        "expertise". The training is performed for a fixed number of steps for
        each agent.
        """
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
        """
        Merges the knowledge of the best-performing agents into the base model.

        This is achieved by averaging the weights (state_dict) of the Long-Term
        Memory (LTM) modules from the provided list of 'best' agents. The resulting
        averaged LTM state is then loaded into the base model, effectively
        assimilating the collective knowledge of the top performers.

        Args:
            best_agents: A list of the top-performing Agent objects.
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

    def _initialize_evaluation(self, tokenizer):
        """
        Initializes the data structures needed for a collaborative evaluation cycle.

        Args:
            tokenizer: The tokenizer instance, used to get special token IDs.

        Returns:
            A tuple containing:
            - A dictionary to store the fitness scores for each agent.
            - A dictionary mapping special action names to their token IDs.
        """
        scores = {agent.agent_id: 0.0 for agent in self.agents}
        tokens = {
            "ask_help": tokenizer.char_to_idx.get("<ASK_FOR_HELP>"),
            "i_dont_know": tokenizer.char_to_idx.get("<I_DONT_KNOW>"),
        }
        return scores, tokens

    def _handle_collaboration_request(self, ctx: CollaborationContext):
        """
        Manages the 'ask for help' scenario in collaborative evaluation.

        The agent that asked for help (proposer) is rewarded. Another agent (helper)
        is chosen to provide a response. This response is then critiqued by all other
        agents. The helper and proposer are both rewarded or penalized based on the
        quality of the help provided, as judged by the critics.

        Args:
            ctx: The context object containing all necessary information for this interaction.
        """
        # Reward the agent for asking for help, as it's a desirable collaborative behavior.
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

        # Get the average critique score for the helper's response
        full_critique_sequence = torch.cat([context, new_helper_tokens], dim=1)
        avg_critique_score = self._batch_critique(
            full_critique_sequence.expand(len(critics), -1), critics
        )

        # Reward or penalize based on the critique.
        if avg_critique_score > self.SUCCESS_THRESHOLD:
            # If the help was good, reward both the helper and the original
            # proposer. This encourages helpers to provide quality answers and
            # proposers to ask good questions that lead to useful outcomes.
            reward = self.REWARD_GOOD_HELP * avg_critique_score
            ctx.scores[helper.agent_id] += reward
            ctx.scores[ctx.proposer.agent_id] += reward
        else:
            # If the help was poor, only the helper is penalized.
            penalty = self.PENALTY_BAD_HELP * (1 - avg_critique_score)
            ctx.scores[helper.agent_id] += penalty

    def _handle_independent_response(self, ctx: IndependentResponseContext):
        """
        Manages the 'independent response' scenario in collaborative evaluation.

        The agent's response is evaluated by all other agents in the population
        (the 'critics'). The proposing agent is then rewarded or penalized based
        on the average critique score, encouraging high-quality, independent solutions.

        Args:
            ctx: The context object containing all necessary information for this interaction.
        """
        # All other agents in the population act as critics.
        critics = (
            [a for a in self.agents if a.agent_id != ctx.proposer.agent_id]
            or [ctx.proposer]  # If no other critics, proposer critiques itself
        )
        new_response_tokens = ctx.response[:, ctx.prompt_tokens.shape[1] :]

        # Prepare for batch critique
        full_sequence = torch.cat([ctx.prompt_tokens, new_response_tokens], dim=1).expand(
            len(critics), -1
        )

        # Get the average critique score
        avg_critique_score = self._batch_critique(full_sequence, critics)

        # Reward or penalize the proposer based on the critique.
        if avg_critique_score > self.SUCCESS_THRESHOLD:
            # Reward successful independent problem-solving.
            reward = self.REWARD_INDEPENDENT_SUCCESS * avg_critique_score
            ctx.scores[ctx.proposer.agent_id] += reward
        else:
            # Penalize unsuccessful or low-quality independent responses.
            penalty = self.PENALTY_INDEPENDENT_FAILURE * (1 - avg_critique_score)
            ctx.scores[ctx.proposer.agent_id] += penalty

    def _batch_critique(self, full_sequence: torch.Tensor, critics: List[Agent]) -> float:
        """
        Performs a batched critique of a response by leveraging the base model.

        This method is architecturally significant because it decouples the AgentManager
        from the internal implementation of the Transformer. Instead of manually
        re-implementing the embedding lookup and LTM state calculation, it prepares
        a batch of LTM states from the critic agents and uses the standard
        `base_model.forward()` method with the `ltm_override` parameter.

        Args:
            full_sequence: The complete token sequence (prompt + response) to be evaluated.
                           Shape: (num_critics, seq_len).
            critics: A list of critic Agent objects.

        Returns:
            The average value score from all critic agents.
        """
        if not critics:
            return 0.0

        with torch.no_grad():
            # Prepare a batch of LTM states from all critic agents.
            # This is more efficient than a Python loop for the forward pass.
            ltm_states = torch.zeros(
                (
                    len(critics),
                    1,
                    self.base_model.config.d_model,
                ),
                device=self.accelerator.device,
                dtype=self.base_model.layers.embedding.embedding.weight.dtype,
            )

            for i, critic in enumerate(critics):
                # Delegate LTM state computation to the agent itself, fully
                # decoupling the manager from the model's internal architecture.
                ltm_states[i] = critic.compute_ltm_state(
                    full_sequence[i].unsqueeze(0)
                )

            # A single, efficient forward pass on the base model using the batch of
            # LTM states. The `ltm_override` mechanism allows us to evaluate all
            # critics in parallel without needing to know the model's internal details.
            _, values, _ = self.base_model.forward(
                full_sequence, ltm_override=ltm_states
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

    def _run_evaluation_cycle(self, evaluation_data, scores, tokens, device):
        """
        Runs the main evaluation loop, creating prompts and orchestrating agent interactions.
        """
        prompt_len = self.base_model.config.max_seq_len // 2
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
        Orchestrates the evaluation of a single prompt.

        For a given prompt, each agent generates a response. This method then
        dispatches the response to the appropriate handler based on its content
        (e.g., independent answer, request for help).
        """
        for i, proposer in enumerate(self.agents):
            response = proposer.generate_response(prompt_tensor)
            response_token_ids = response[0].tolist()

            if tokens["ask_help"] in response_token_ids:
                # The agent chose to ask for help.
                ctx = CollaborationContext(
                    proposer=proposer,
                    prompt_tokens=prompt_tensor,
                    scores=scores,
                    proposer_index=i,
                    response=response,
                )
                self._handle_collaboration_request(ctx)
            elif tokens["i_dont_know"] in response_token_ids:
                # The agent admitted it doesn't know the answer.
                scores[proposer.agent_id] += self.REWARD_ADMIT_IGNORANCE
            else:
                # The agent provided an independent response.
                ctx = IndependentResponseContext(
                    proposer=proposer,
                    prompt_tokens=prompt_tensor,
                    scores=scores,
                    response=response,
                )
                self._handle_independent_response(ctx)

    def collaborative_evaluation(self, evaluation_data, tokenizer, top_k, device):
        """
        Performs a collaborative evaluation and returns the top-performing agents.

        This method implements a nuanced, peer-review-based scoring system that
        rewards desirable collaborative behaviors. Agents are presented with prompts
        and can choose to respond independently, ask for help, or admit ignorance.
        The quality of these actions is judged by the other agents in the population
        (the "critics"), leading to a fitness score for each agent.

        Returns:
            A list of the `top_k` best-performing agents based on fitness scores.
        """
        if not self.agents or len(self.agents) < 2:
            logging.warning("Collaborative evaluation requires at least 2 agents.")
            return self.agents[:top_k]

        scores, tokens = self._initialize_evaluation(tokenizer)
        self._run_evaluation_cycle(evaluation_data, scores, tokens, device)
        return self._finalize_evaluation(scores, top_k)
