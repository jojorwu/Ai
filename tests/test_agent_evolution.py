"""
Integration test for the agent-based evolutionary training stage.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from model import Transformer
from agent_manager import AgentManager
from tokenizer import Tokenizer

class TestAgentEvolution(unittest.TestCase):
    """
    Tests the agent evolution training stage (Stage 4).
    """
    def setUp(self):
        """Set up a small model and a dummy data directory for the test."""
        # Create a dummy data directory
        self.data_dir = "temp_agent_test_data"
        os.makedirs(self.data_dir, exist_ok=True)
        with open(os.path.join(self.data_dir, "test1.txt"), "w") as f:
            f.write("This is the first test document for agent specialization.")
        with open(os.path.join(self.data_dir, "test2.txt"), "w") as f:
            f.write("This is the second test document, full of novel information.")

    def tearDown(self):
        """Clean up the dummy data directory."""
        import shutil
        shutil.rmtree(self.data_dir)

    def test_collaborative_evaluation_scenarios(self):
        """
        Tests the nuanced scoring of the collaborative_evaluation method.
        """
        print("\\nRunning Test: Collaborative Evaluation Scenarios...")

        config = Config.from_json('config.json')
        config.model.d_model, config.model.num_layers, config.model.num_heads, config.model.d_ff = 16, 1, 2, 32
        config.model.ltm_d_hidden, config.model.ltm_num_layers = 8, 1

        tokenizer = Tokenizer(self.data_dir)
        base_model = Transformer(vocab_size=tokenizer.vocab_size, model_config=config.model, ltm_config=config.ltm)

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
                    'agent_0': {'response': [tokenizer.char_to_idx['<ASK_FOR_HELP>']], 'critique_score': 0.9}, # Proposer
                    'agent_1': {'response': [1, 2], 'critique_score': 0.9} # Helper
                }, "expected_winner": "agent_1" # Helper also succeeds independently, so it gets more points
            },
            "RECEIVES_BAD_HELP": {
                "agents_setup": {
                    'agent_0': {'response': [tokenizer.char_to_idx['<ASK_FOR_HELP>']], 'critique_score': 0.2}, # Proposer
                    'agent_1': {'response': [1, 2], 'critique_score': 0.2} # Helper
                }, "expected_winner": "agent_0" # Proposer gets small reward, helper is penalized
            }
        }

        import types
        for scenario_name, details in scenarios.items():
            with self.subTest(scenario=scenario_name):
                agent_manager = AgentManager(base_model=base_model, num_agents=2)

                def mock_generate_response(agent_self, prompt, image_data=None, max_len=50):
                    return details['agents_setup'][agent_self.agent_id]['response']

                def mock_critique_response(agent_self, prompt, image_data, response):
                    # In the HELP scenarios, the response being critiqued is always from the helper (agent_1)
                    if details['agents_setup']['agent_0']['response'][0] == tokenizer.char_to_idx['<ASK_FOR_HELP>']:
                        return details['agents_setup']['agent_1']['critique_score']

                    # In independent scenarios, find which agent generated this response
                    for agent_id, setup in details['agents_setup'].items():
                        if setup['response'] == response:
                            return setup['critique_score']
                    return 0.0

                for i, agent in enumerate(agent_manager.agents):
                    agent.generate_response = types.MethodType(mock_generate_response, agent)
                    agent.critique_response = types.MethodType(mock_critique_response, agent)

                best_agents = agent_manager.collaborative_evaluation(
                    evaluation_data=[([1, 2, 3], None)],
                    tokenizer=tokenizer,
                    top_k=1
                )

                self.assertEqual(best_agents[0].agent_id, details["expected_winner"])
                print(f"  - Scenario '{scenario_name}' PASSED.")

    def test_merge_agents_functionality(self):
        """
        Tests that the merge_agents method correctly updates the base model's LTM.
        """
        print("\\nRunning Test: Agent Merging Functionality...")
        config = Config.from_json('config.json')
        config.model.d_model, config.model.num_layers, config.model.num_heads, config.model.d_ff = 16, 1, 2, 32
        config.model.ltm_d_hidden, config.model.ltm_num_layers = 8, 1

        tokenizer = Tokenizer(self.data_dir)
        base_model = Transformer(vocab_size=tokenizer.vocab_size, model_config=config.model, ltm_config=config.ltm)
        initial_ltm_state = base_model.long_term_memory.get_state()
        initial_ltm_weights = initial_ltm_state['linear_0']['W']

        agent_manager = AgentManager(base_model=base_model, num_agents=2)

        # Manually create a "best agent" and modify its LTM weights
        best_agent = agent_manager.agents[0]
        winning_agent_ltm = best_agent.model.long_term_memory
        changed_weights = np.copy(winning_agent_ltm.layers[0].W)
        changed_weights += 0.5  # Introduce a noticeable change
        winning_agent_ltm.layers[0].W = changed_weights

        agent_manager.merge_agents([best_agent])

        updated_ltm_state = base_model.long_term_memory.get_state()
        updated_ltm_weights = updated_ltm_state['linear_0']['W']

        self.assertFalse(np.allclose(initial_ltm_weights, updated_ltm_weights),
                         "Base model's LTM weights did not change after merge.")
        self.assertTrue(np.allclose(updated_ltm_weights, changed_weights),
                        "Merged LTM weights do not match the winning agent's weights.")
        print("Agent Merging Functionality test PASSED.")

if __name__ == "__main__":
    unittest.main()
