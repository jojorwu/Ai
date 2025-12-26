"""
Integration test for the agent-based evolutionary training stage.
"""
import logging
import os
import shutil
import types
import unittest

import numpy as np

from agent_manager import AgentManager
from config import Config
from model import Transformer
from tokenizer import Tokenizer


class TestAgentEvolution(unittest.TestCase):
    """
    Tests the agent evolution training stage.
    """

    def setUp(self):
        """Set up a small model and a dummy data directory for the test."""
        self.data_dir = "temp_agent_test_data"
        os.makedirs(self.data_dir, exist_ok=True)
        with open(os.path.join(self.data_dir, "test1.txt"), "w", encoding='utf-8') as f:
            f.write("This is the first test document for agent specialization.")
        with open(os.path.join(self.data_dir, "test2.txt"), "w", encoding='utf-8') as f:
            f.write("This is the second test document, full of novel information.")

    def tearDown(self):
        """Clean up the dummy data directory."""
        shutil.rmtree(self.data_dir)

    def test_collaborative_evaluation_scenarios(self):
        """
        Tests the nuanced scoring of the collaborative_evaluation method.
        """
        logging.info("\nRunning Test: Collaborative Evaluation Scenarios...")

        config = Config.from_json('config.json')
        config.model.d_model, config.model.num_layers, config.model.num_heads, \
            config.model.d_ff = 16, 1, 2, 32
        config.model.ltm_d_hidden, config.model.ltm_num_layers = 8, 1

        tokenizer = Tokenizer(self.data_dir)
        base_model = Transformer(vocab_size=tokenizer.vocab_size,
                                 model_config=config.model,
                                 vision_config=config.vision,
                                 ltm_config=config.ltm)

        scenarios = {
            "INDEPENDENT_SUCCESS": {
                "agents_setup": {
                    'agent_0': {'response': [10, 11], 'critique_score': 0.9},
                    'agent_1': {'response': [12, 13], 'critique_score': 0.2}
                }, "expected_winner": "agent_0"
            },
            "INDEPENDENT_FAILURE": {
                "agents_setup": {
                    'agent_0': {'response': [10, 11], 'critique_score': 0.1},
                    'agent_1': {'response': [12, 13], 'critique_score': 0.8}
                }, "expected_winner": "agent_1"
            },
            "RECEIVES_GOOD_HELP": {
                "agents_setup": {
                    'agent_0': {'response': [tokenizer.char_to_idx['<ASK_FOR_HELP>']],
                                'critique_score': 0.9},
                    'agent_1': {'response': [1, 2], 'critique_score': 0.9}
                }, "expected_winner": "agent_1"
            },
            "RECEIVES_BAD_HELP": {
                "agents_setup": {
                    'agent_0': {'response': [tokenizer.char_to_idx['<ASK_FOR_HELP>']],
                                'critique_score': 0.2},
                    'agent_1': {'response': [1, 2], 'critique_score': 0.2}
                }, "expected_winner": "agent_0"
            }
        }

        for scenario_name, details in scenarios.items():
            with self.subTest(scenario=scenario_name):
                agent_manager = AgentManager(base_model=base_model, num_agents=2)

                def mock_generate(model_self, inputs, details=details,
                                  agent_manager=agent_manager):
                    agent_id = next(agent.agent_id for agent in agent_manager.agents
                                    if agent.model is model_self)
                    return np.array(details['agents_setup'][agent_id]['response'])

                def mock_critique_response(_, __, ___, response, details=details):
                    if (details['agents_setup']['agent_0']['response'][0]
                            == tokenizer.char_to_idx['<ASK_FOR_HELP>']):
                        return details['agents_setup']['agent_1']['critique_score']

                    for setup in details['agents_setup'].values():
                        if np.array_equal(setup['response'], response):
                            return setup['critique_score']
                    return 0.0

                for agent in agent_manager.agents:
                    agent.model.generate = types.MethodType(mock_generate, agent.model)
                    agent.critique_response = types.MethodType(mock_critique_response, agent)

                best_agents = agent_manager.collaborative_evaluation(
                    evaluation_data=[([1, 2, 3], None)],
                    tokenizer=tokenizer,
                    top_k=1
                )

                self.assertEqual(best_agents[0].agent_id, details["expected_winner"])
                logging.info("  - Scenario '%s' PASSED.", scenario_name)

    def test_merge_agents_functionality(self):
        """
        Tests that the merge_agents method correctly updates the base model's LTM.
        """
        logging.info("\nRunning Test: Agent Merging Functionality...")
        config = Config.from_json('config.json')
        config.model.d_model, config.model.num_layers, config.model.num_heads, \
            config.model.d_ff = 16, 1, 2, 32
        config.model.ltm_d_hidden, config.model.ltm_num_layers = 8, 1

        tokenizer = Tokenizer(self.data_dir)
        base_model = Transformer(vocab_size=tokenizer.vocab_size,
                                 model_config=config.model,
                                 vision_config=config.vision,
                                 ltm_config=config.ltm)
        initial_ltm_state = base_model.long_term_memory.get_state()
        initial_ltm_weights = initial_ltm_state['linear_0']['weights']

        agent_manager = AgentManager(base_model=base_model, num_agents=2)

        best_agent = agent_manager.agents[0]
        winning_agent_ltm = best_agent.model.long_term_memory
        changed_weights = np.copy(winning_agent_ltm.layers[0].weights)
        changed_weights += 0.5
        winning_agent_ltm.layers[0].weights = changed_weights

        agent_manager.merge_agents([best_agent])

        updated_ltm_state = base_model.long_term_memory.get_state()
        updated_ltm_weights = updated_ltm_state['linear_0']['weights']

        self.assertFalse(np.allclose(initial_ltm_weights, updated_ltm_weights),
                         "Base model's LTM weights did not change after merge.")
        self.assertTrue(np.allclose(updated_ltm_weights, changed_weights),
                        "Merged LTM weights do not match the winning agent's weights.")
        logging.info("Agent Merging Functionality test PASSED.")


if __name__ == "__main__":
    unittest.main()
