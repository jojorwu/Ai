"""
Implementation of the AgentManager class for managing the agent lifecycle.
"""
import copy
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
    x = np.array(data[:end_idx], dtype=np.int64).reshape(batch_size, -1)
    y = np.array(data[1:end_idx+1], dtype=np.int64).reshape(batch_size, -1)

    for i in range(0, x.shape[1], seq_len):
        yield x[:, i:i+seq_len], y[:, i:i+seq_len]


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
        print(f"Cloning {self.num_agents} agents from the base model...")
        for i in range(self.num_agents):
            agent = Agent(self.base_model, agent_id=f"agent_{i}")
            self.agents.append(agent)
        print("Agents cloned successfully.")

    def specialize_agents_on_dataset(self, full_data: list, tokenizer: Tokenizer, seq_len: int, batch_size: int, steps_per_agent: int):
        """
        Conducts a "specialization" phase where each agent is trained
        on a unique subset of the data.
        """
        if not full_data:
            print("Warning: No data provided for agent specialization.")
            return

        # Split the entire dataset into equal parts for each agent
        data_chunks = np.array_split(full_data, self.num_agents)

        print("Specializing agents on different data subsets...")
        for i, agent in enumerate(self.agents):
            # The chunk is already a list of (tokens, image_data) tuples
            agent_data = data_chunks[i]
            if len(agent_data) == 0:
                print(f"  - Skipping {agent.agent_id}, no data assigned.")
                continue

            # For now, we only specialize on the text part.
            # TODO: Update agent experience to handle images.
            agent_text_data = [item[0] for item in agent_data]
            # Flatten the list of token lists into a single list
            flat_token_list = [token for sublist in agent_text_data for token in sublist]


            print(f"  - Specializing {agent.agent_id} on {len(flat_token_list)} tokens...")

            batch_generator = get_agent_batches(flat_token_list, batch_size, seq_len)

            steps_done = 0
            for x_batch, y_batch in batch_generator:
                if steps_done >= steps_per_agent:
                    break
                agent.experience(x_batch, y_batch)
                steps_done += 1

            if steps_done < steps_per_agent:
                print(f"    - Warning: Only {steps_done}/{steps_per_agent} steps were performed for {agent.agent_id} due to insufficient data.")

        print("Agent specialization complete.")

    def evaluate_and_select_best(self, top_k: int) -> List[Agent]:
        """
        Evaluates all agents based on their internally tracked fitness and returns the top_k best.
        """
        print(f"Evaluating agents and selecting top {top_k}...")
        if not self.agents:
            return []

        sorted_agents = sorted(self.agents, key=lambda a: a.get_fitness_score(), reverse=True)

        top_agents = sorted_agents[:top_k]
        print("  - Fitness scores:")
        for agent in sorted_agents:
             print(f"    - {agent.agent_id}: {agent.get_fitness_score():.4f}")

        return top_agents

    def merge_agents(self, best_agents: List[Agent]):
        """
        Averages the LTM weights of the "best" agents and updates the base model's LTM.
        """
        if not best_agents:
            print("Warning: No best agents to merge.")
            return

        print(f"Merging LTM weights from {len(best_agents)} best agents...")

        ltm_states = [agent.get_ltm_state() for agent in best_agents if agent.get_ltm_state() is not None]
        if not ltm_states:
            print("Warning: None of the best agents had a valid LTM state.")
            return

        avg_ltm_state = copy.deepcopy(ltm_states[0])

        for layer_name in avg_ltm_state:
            sum_W = sum(state[layer_name]['W'] for state in ltm_states)
            sum_b = sum(state[layer_name].get('b', 0) for state in ltm_states)

            avg_ltm_state[layer_name]['W'] = sum_W / len(ltm_states)
            if 'b' in avg_ltm_state[layer_name]:
                avg_ltm_state[layer_name]['b'] = sum_b / len(ltm_states)

        if self.base_model.long_term_memory:
            self.base_model.long_term_memory.set_state(avg_ltm_state)
            print("Base model's LTM has been updated with merged weights.")

    def cross_critique_evaluation(self, evaluation_data: list, tokenizer: Tokenizer, top_k: int) -> List[Agent]:
        """
        Evaluates agents using a cross-critique method.
        Each agent generates a response, and the others critique it.
        """
        if not self.agents or len(self.agents) < 2:
            print("Warning: Cross-critique requires at least 2 agents.")
            return self.agents[:top_k]

        agent_scores = {agent.agent_id: [] for agent in self.agents}

        eval_prompts = evaluation_data[:min(len(evaluation_data), self.num_agents * 2)]

        for prompt_data in eval_prompts:
            prompt_tokens, prompt_image = prompt_data

            for i, proposer in enumerate(self.agents):
                response_tokens = proposer.generate_response(prompt_tokens, prompt_image)
                critique_scores = []
                critics = self.agents[:i] + self.agents[i+1:]
                for critic in critics:
                    score = critic.critique_response(prompt_tokens, prompt_image, response_tokens)
                    critique_scores.append(score)

                avg_score = np.mean(critique_scores) if critique_scores else 0
                agent_scores[proposer.agent_id].append(avg_score)

        for agent in self.agents:
            final_score = np.mean(agent_scores[agent.agent_id]) if agent_scores[agent.agent_id] else 0
            agent.update_fitness_score(final_score)

        sorted_agents = sorted(self.agents, key=lambda a: a.get_fitness_score(), reverse=True)

        print("  - Cross-Critique Fitness scores:")
        for agent in sorted_agents:
             print(f"    - {agent.agent_id}: {agent.get_fitness_score():.4f}")

        return sorted_agents[:top_k]
