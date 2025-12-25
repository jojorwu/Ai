"""
Implementation of the AgentManager class for managing the agent lifecycle.
"""
import copy
import logging
from typing import List

from backend import np

from agent import Agent
from model import Transformer
from tokenizer import Tokenizer


def get_agent_batches(data, batch_size, seq_len):
    """
    Simplified batch generator for agent specialization.
    Unlike the main one, it doesn't shuffle and works with a single data chunk.
    """
    num_total_tokens = len(data)
    if num_total_tokens < seq_len:
        return

    num_sequences = (num_total_tokens - 1) // seq_len
    if num_sequences < batch_size:
        return

    num_batches = num_sequences // batch_size
    if num_batches == 0:
        return

    end_idx = num_batches * batch_size * seq_len
    x = np.array([item[0] for item in data[:end_idx]], dtype=np.int64).reshape(batch_size, -1)
    y = np.array([item[0] for item in data[1:end_idx + 1]], dtype=np.int64).reshape(batch_size, -1)
    images = np.array([item[1] for item in data[:end_idx]], dtype=object).reshape(batch_size, -1)

    for i in range(0, x.shape[1], seq_len):
        yield x[:, i:i + seq_len], y[:, i:i + seq_len], images[:, i:i + seq_len]


class AgentManager:
    """
    Manages the creation, specialization, and evaluation of a population of agents.
    """

    def __init__(self, base_model: Transformer, num_agents: int):
        self.base_model = base_model
        self.num_agents = num_agents
        self.agents: List[Agent] = []
        self.fork_agents()

    def fork_agents(self):
        """Creates (clones) a population of agents from the base model."""
        logging.info(f"Cloning {self.num_agents} agents from the base model...")
        for i in range(self.num_agents):
            agent = Agent(self.base_model, agent_id=f"agent_{i}")
            self.agents.append(agent)
        logging.info("Agents cloned successfully.")

    def specialize_agents_on_dataset(self, full_data: list, tokenizer: Tokenizer,
                                     seq_len: int, batch_size: int, steps_per_agent: int):
        """
        Conducts a "specialization" phase where each agent is trained
        on a unique subset of the data.
        """
        if not full_data:
            logging.warning("No data provided for agent specialization.")
            return

        data_chunks = np.array_split(full_data, self.num_agents)

        logging.info("Specializing agents on different data subsets...")
        for i, agent in enumerate(self.agents):
            agent_data = data_chunks[i]
            if len(agent_data) == 0:
                logging.info(f"  - Skipping {agent.agent_id}, no data assigned.")
                continue

            logging.info(f"  - Specializing {agent.agent_id} on {len(agent_data)} items...")

            batch_generator = get_agent_batches(agent_data, batch_size, seq_len)

            steps_done = 0
            for x_batch, y_batch, image_batch in batch_generator:
                if steps_done >= steps_per_agent:
                    break
                agent.experience(x_batch, y_batch, image_batch)
                steps_done += 1

            if steps_done < steps_per_agent:
                logging.warning(f"    - Only {steps_done}/{steps_per_agent} steps were "
                                f"performed for {agent.agent_id} due to insufficient data.")

        logging.info("Agent specialization complete.")

    def merge_agents(self, best_agents: List[Agent]):
        """
        Averages the LTM weights of the "best" agents and updates the base model's LTM.
        """
        if not best_agents:
            logging.warning("No best agents to merge.")
            return

        logging.info(f"Merging LTM weights from {len(best_agents)} best agents...")

        ltm_states = [agent.get_ltm_state() for agent in best_agents if agent.get_ltm_state() is not None]
        if not ltm_states:
            logging.warning("None of the best agents had a valid LTM state.")
            return

        avg_ltm_state = copy.deepcopy(ltm_states[0])

        for layer_name in avg_ltm_state:
            sum_w = sum(state[layer_name]['weights'] for state in ltm_states)
            sum_b = sum(state[layer_name].get('bias', 0) for state in ltm_states)

            avg_ltm_state[layer_name]['weights'] = sum_w / len(ltm_states)
            if 'bias' in avg_ltm_state[layer_name]:
                avg_ltm_state[layer_name]['bias'] = sum_b / len(ltm_states)

        if self.base_model.long_term_memory:
            self.base_model.long_term_memory.set_state(avg_ltm_state)
            logging.info("Base model's LTM has been updated with merged weights.")

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def collaborative_evaluation(self, evaluation_data: list,
                                 tokenizer: Tokenizer, top_k: int) -> List[Agent]:
        """
        Evaluates agents with a nuanced scoring system that rewards independent success
        and penalizes failure and bad help, based on peer critique.
        """
        if not self.agents or len(self.agents) < 2:
            logging.warning("Collaborative evaluation requires at least 2 agents.")
            return self.agents[:top_k]

        agent_scores = {agent.agent_id: 0.0 for agent in self.agents}
        reward_independent_success = 5.0
        penalty_independent_failure = -5.0
        reward_good_help = 3.0
        penalty_bad_help = -3.0
        reward_asking_for_help = 0.5
        reward_admit_ignorance = 1.0
        success_threshold = 0.5

        ask_help_token_id = tokenizer.char_to_idx.get('<ASK_FOR_HELP>')
        i_dont_know_token_id = tokenizer.char_to_idx.get('<I_DONT_KNOW>')

        eval_prompts = evaluation_data[:min(len(evaluation_data), self.num_agents * 2)]

        for prompt_data in eval_prompts:
            prompt_tokens, prompt_image = prompt_data
            for i, proposer in enumerate(self.agents):
                critics = [a for a in self.agents if a.agent_id != proposer.agent_id]
                proposer_response = proposer.generate_response(np.array(prompt_tokens), prompt_image)

                if ask_help_token_id in proposer_response:
                    agent_scores[proposer.agent_id] += reward_asking_for_help
                    helper = self.agents[(i + 1) % len(self.agents)]
                    if helper.agent_id == proposer.agent_id:
                        continue

                    context = list(prompt_tokens) + proposer_response
                    helper_response = helper.generate_response(np.array(context), prompt_image)

                    critics_for_helper = [a for a in self.agents if
                                          a.agent_id not in [proposer.agent_id, helper.agent_id]]
                    if not critics_for_helper:
                        critics_for_helper = [proposer]

                    critique_scores = [c.critique_response(context, prompt_image, helper_response)
                                       for c in critics_for_helper]
                    avg_critique_score = np.mean(critique_scores) if critique_scores else 0

                    if avg_critique_score > success_threshold:
                        agent_scores[helper.agent_id] += reward_good_help * avg_critique_score
                        agent_scores[proposer.agent_id] += reward_good_help * avg_critique_score
                    else:
                        agent_scores[helper.agent_id] += penalty_bad_help * (1 - avg_critique_score)

                elif i_dont_know_token_id in proposer_response:
                    agent_scores[proposer.agent_id] += reward_admit_ignorance

                else:
                    critique_scores = [c.critique_response(prompt_tokens, prompt_image,
                                                           proposer_response) for c in critics]
                    avg_critique_score = np.mean(critique_scores) if critique_scores else 0

                    if avg_critique_score > success_threshold:
                        agent_scores[proposer.agent_id] += reward_independent_success * avg_critique_score
                    else:
                        agent_scores[proposer.agent_id] += penalty_independent_failure * (1 - avg_critique_score)

        for agent in self.agents:
            agent.update_fitness_score(agent_scores[agent.agent_id])

        sorted_agents = sorted(self.agents, key=lambda a: a.get_fitness_score(), reverse=True)
        logging.info("  - Collaborative Evaluation Fitness scores:")
        for agent in sorted_agents:
            logging.info(f"    - {agent.agent_id}: {agent.get_fitness_score():.4f}")
        return sorted_agents[:top_k]
