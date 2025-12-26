"""
Implementation of the AgentManager class for managing the agent lifecycle.
"""
import copy
import logging
from dataclasses import dataclass
from typing import List

from agent import Agent
from backend import np
from model import Transformer


@dataclass
class SpecializationConfig:
    """Configuration for the agent specialization process."""
    full_data: list
    seq_len: int
    batch_size: int
    steps_per_agent: int


def get_agent_batches(data, batch_size, seq_len):
    """
    Simplified batch generator for agent specialization.
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
        logging.info("Cloning %d agents from the base model...", self.num_agents)
        for i in range(self.num_agents):
            agent = Agent(self.base_model, agent_id=f"agent_{i}")
            self.agents.append(agent)
        logging.info("Agents cloned successfully.")

    def specialize_agents_on_dataset(self, spec_config: SpecializationConfig):
        """
        Conducts a "specialization" phase where each agent is trained
        on a unique subset of the data.
        """
        if not spec_config.full_data:
            logging.warning("No data provided for agent specialization.")
            return

        data_chunks = np.array_split(spec_config.full_data, self.num_agents)

        logging.info("Specializing agents on different data subsets...")
        for i, agent in enumerate(self.agents):
            agent_data = data_chunks[i]
            if len(agent_data) == 0:
                logging.info("  - Skipping %s, no data assigned.", agent.agent_id)
                continue

            logging.info("  - Specializing %s on %d items...", agent.agent_id, len(agent_data))

            batch_generator = get_agent_batches(agent_data, spec_config.batch_size, spec_config.seq_len)

            steps_done = 0
            for x_batch, y_batch, image_batch in batch_generator:
                if steps_done >= spec_config.steps_per_agent:
                    break
                agent.experience(x_batch, y_batch, image_batch)
                steps_done += 1

            if steps_done < spec_config.steps_per_agent:
                logging.warning("    - Only %d/%d steps were performed for %s due to insufficient data.",
                                steps_done, spec_config.steps_per_agent, agent.agent_id)

        logging.info("Agent specialization complete.")

    def merge_agents(self, best_agents: List[Agent]):
        """
        Averages the LTM weights of the "best" agents and updates the base model's LTM.
        """
        if not best_agents:
            logging.warning("No best agents to merge.")
            return

        logging.info("Merging LTM weights from %d best agents...", len(best_agents))

        ltm_states = [agent.get_ltm_state() for agent in best_agents if agent.get_ltm_state() is not None]
        if not ltm_states:
            logging.warning("None of the best agents had a valid LTM state.")
            return

        avg_ltm_state = copy.deepcopy(ltm_states[0])
        avg_ltm_state['memory_state'] = np.zeros_like(avg_ltm_state['memory_state'])
        for layer_name in avg_ltm_state['children']:
            avg_ltm_state['children'][layer_name]['weights'] = np.zeros_like(
                avg_ltm_state['children'][layer_name]['weights'])
            if 'bias' in avg_ltm_state['children'][layer_name]:
                avg_ltm_state['children'][layer_name]['bias'] = np.zeros_like(
                    avg_ltm_state['children'][layer_name]['bias'])

        for state in ltm_states:
            avg_ltm_state['memory_state'] += state['memory_state']
            for layer_name, layer_state in state['children'].items():
                avg_ltm_state['children'][layer_name]['weights'] += layer_state['weights']
                if 'bias' in layer_state:
                    avg_ltm_state['children'][layer_name]['bias'] += layer_state.get('bias', 0)

        num_best_agents = len(ltm_states)
        avg_ltm_state['memory_state'] /= num_best_agents
        for layer_name in avg_ltm_state['children']:
            avg_ltm_state['children'][layer_name]['weights'] /= num_best_agents
            if 'bias' in avg_ltm_state['children'][layer_name]:
                avg_ltm_state['children'][layer_name]['bias'] /= num_best_agents

        if self.base_model.long_term_memory:
            self.base_model.long_term_memory.set_state(avg_ltm_state)
            logging.info("Base model's LTM has been updated with merged weights.")

    def _initialize_evaluation(self, tokenizer):
        """Initializes scores and special tokens for evaluation."""
        scores = {agent.agent_id: 0.0 for agent in self.agents}
        tokens = {
            "ask_help": tokenizer.char_to_idx.get('<ASK_FOR_HELP>'),
            "i_dont_know": tokenizer.char_to_idx.get('<I_DONT_KNOW>')
        }
        return scores, tokens

    def _handle_collaboration_request(self, proposer, prompt_tokens, prompt_image, scores, i, response):
        """Handles the scenario where an agent asks for help."""
        scores[proposer.agent_id] += 0.5
        helper = self.agents[(i + 1) % len(self.agents)]
        if helper.agent_id == proposer.agent_id:
            return

        context = list(prompt_tokens) + response.tolist()
        helper_response = helper.generate_response(np.array(context), prompt_image)

        critics = [a for a in self.agents if a.agent_id not in [proposer.agent_id, helper.agent_id]] or [proposer]
        critique_scores = [c.critique_response(context, prompt_image, helper_response) for c in critics]
        avg_critique_score = np.mean(critique_scores) if critique_scores else 0

        if avg_critique_score > 0.5:
            scores[helper.agent_id] += 3.0 * avg_critique_score
            scores[proposer.agent_id] += 3.0 * avg_critique_score
        else:
            scores[helper.agent_id] += -3.0 * (1 - avg_critique_score)

    def _handle_independent_response(self, proposer, prompt_tokens, prompt_image, scores, response):
        """Handles the scenario where an agent responds independently."""
        critics = [a for a in self.agents if a.agent_id != proposer.agent_id]
        critique_scores = [c.critique_response(prompt_tokens, prompt_image, response) for c in critics]
        avg_critique_score = np.mean(critique_scores) if critique_scores else 0

        if avg_critique_score > 0.5:
            scores[proposer.agent_id] += 5.0 * avg_critique_score
        else:
            scores[proposer.agent_id] += -5.0 * (1 - avg_critique_score)

    def _finalize_evaluation(self, scores, top_k):
        """Finalizes evaluation by updating and sorting agents."""
        for agent in self.agents:
            agent.update_fitness_score(scores[agent.agent_id])
        sorted_agents = sorted(self.agents, key=lambda a: a.get_fitness_score(), reverse=True)
        logging.info("  - Collaborative Evaluation Fitness scores:")
        for agent in sorted_agents:
            logging.info("    - %s: %.4f", agent.agent_id, agent.get_fitness_score())
        return sorted_agents[:top_k]

    def collaborative_evaluation(self, evaluation_data, tokenizer, top_k):
        """
        Evaluates agents with a nuanced scoring system.
        """
        if not self.agents or len(self.agents) < 2:
            logging.warning("Collaborative evaluation requires at least 2 agents.")
            return self.agents[:top_k]

        scores, tokens = self._initialize_evaluation(tokenizer)
        eval_prompts = evaluation_data[:min(len(evaluation_data), self.num_agents * 2)]

        for prompt_data in eval_prompts:
            prompt_tokens, prompt_image = prompt_data
            for i, proposer in enumerate(self.agents):
                response = proposer.generate_response(np.array(prompt_tokens), prompt_image)

                if tokens["ask_help"] in response:
                    self._handle_collaboration_request(proposer, prompt_tokens, prompt_image, scores, i, response)
                elif tokens["i_dont_know"] in response:
                    scores[proposer.agent_id] += 1.0
                else:
                    self._handle_independent_response(proposer, prompt_tokens, prompt_image, scores, response)

        return self._finalize_evaluation(scores, top_k)
