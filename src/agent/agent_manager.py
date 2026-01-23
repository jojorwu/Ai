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


@dataclass
class CritiqueScoringContext:
    """Context for applying critique-based scores."""
    scores: dict
    avg_critique_score: float
    success_reward: float
    failure_penalty: float
    agents_to_reward: List[Agent]
    agents_to_penalize: List[Agent]


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
        self.fork_agents()

    def fork_agents(self):
        """
        Creates a population of agents by cloning the base model.

        Each agent shares the weights of the base model but will have its own
        unique Long-Term Memory (LTM) module. This method populates the
        `self.agents` list.
        """
        logging.info("Cloning %d agents from the base model...", self.num_agents)
        for i in range(self.num_agents):
            agent = Agent(self.base_model, agent_id=f"agent_{i}")
            self.agents.append(agent)
        logging.info("Agents cloned successfully.")

    def specialize_agents_on_dataset(self, spec_config: SpecializationConfig, device):
        """
        Conducts a "specialization" phase where each agent is trained on a
        unique subset of the data.

        This process allows each agent to develop a specialized Long-Term Memory (LTM)
        based on its unique experiences, fostering diversity in the agent population.
        The training is done for a fixed number of steps per agent, and only the
        LTM weights are updated.

        Args:
            spec_config: A dataclass containing the configuration for specialization,
                         including the full dataset and training parameters like
                         batch size and steps per agent.
            device: The device (e.g., 'cuda' or 'cpu') to perform the training on.
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

    def _get_critics(self, proposer: Agent, agents_to_exclude: List[Agent]) -> List[Agent]:
        """
        Selects agents from the population to act as critics.

        This method returns a list of agents that are not in the `agents_to_exclude`
        list. If no such agents are available, it returns a list containing only
        the `proposer` as a fallback.

        Args:
            proposer: The agent who initiated the action, used as a fallback critic.
            agents_to_exclude: A list of agents to exclude from the critic pool.

        Returns:
            A list of critic agents.
        """
        exclude_ids = {agent.agent_id for agent in agents_to_exclude}
        critics = [
            agent for agent in self.agents if agent.agent_id not in exclude_ids
        ]
        # As a fallback, if no other critics are available, the proposer critiques itself.
        return critics or [proposer]

    def _apply_critique_based_scores(self, csc: CritiqueScoringContext):
        """
        Applies rewards or penalties to agents based on a critique score.

        If the critique score is above a success threshold, a scaled reward is given
        to the agents to reward. Otherwise, a scaled penalty is applied to the
        agents to penalize.

        Args:
            csc: The context object containing all scoring parameters.
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
            # Step 1: Compute token embeddings for the entire batch of sequences.
            # The embedding output is scaled by sqrt(d_model) as is standard.
            h = self.base_model.layers.embedding(full_sequence) * math.sqrt(
                self.base_model.config.model.d_model
            )

            # Step 2: Pre-allocate a tensor for the LTM states. This allows us to
            # build the batch of LTM states efficiently.
            ltm_states = torch.zeros(
                (len(critic_ltms), 1, h.size(2)), device=h.device, dtype=h.dtype
            )

            # Step 3: Iterate through each critic's LTM to compute its unique
            # LTM state based on the sequence. This is necessary because each
            # agent has a different LTM.
            for i, ltm in enumerate(critic_ltms):
                if ltm:
                    # The LTM state is computed from the mean of the sequence embeddings,
                    # providing a compressed summary for the critic.
                    ltm_input = h[i].mean(dim=0, keepdim=True).unsqueeze(0)
                    ltm_states[i], _ = ltm(ltm_input)

            # Step 4: Perform a single, batched forward pass on the base model.
            # The `ltm_state` argument is a batch of LTM states, one for each
            # critic. This is highly efficient as it avoids looping and running
            # the model for each critic individually.
            _, values, _ = self.base_model.forward(
                full_sequence, ltm_state=ltm_states
            )
        # Step 5: The final critique score is the average of the value predictions
        # from all critics in the batch.
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
