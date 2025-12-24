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

    def test_agent_evolution_cycle(self):
        """
        Tests a single, complete cycle of agent evolution:
        fork -> specialize -> evaluate -> merge.
        """
        print("\\nRunning Test: Agent Evolution Cycle...")

        # 1. --- Config and Initialization ---
        config = Config.from_json('config.json')
        # Use a smaller model for the test
        config.model.d_model = 16
        config.model.num_layers = 1
        config.model.num_heads = 2
        config.model.d_ff = 32
        config.model.ltm_d_hidden = 8
        config.model.ltm_num_layers = 1

        tokenizer = Tokenizer(self.data_dir)
        base_model = Transformer(vocab_size=tokenizer.vocab_size, model_config=config.model, ltm_config=config.ltm)

        # Store the initial LTM state
        initial_ltm_state = base_model.long_term_memory.get_state()
        initial_ltm_weights = initial_ltm_state['linear_0']['W']

        # 2. --- Agent Evolution Cycle ---
        agent_manager = AgentManager(base_model=base_model, num_agents=2)

        # Mock the generate_response and critique_response methods
        import types
        def mock_generate_response(agent_self, prompt, image_data=None, max_len=50):
            # Agent 0 asks for help
            if agent_self.agent_id == 'agent_0':
                return [tokenizer.char_to_idx['<ASK_FOR_HELP>']]
            # Agent 1 provides help
            else:
                return [tokenizer.char_to_idx['<PROVIDE_HELP>']]

        def mock_critique_response(agent_self, prompt, image_data, response):
            return 1.0

        for agent in agent_manager.agents:
            agent.generate_response = types.MethodType(mock_generate_response, agent)
            agent.critique_response = types.MethodType(mock_critique_response, agent)

        # Also, let's manually change the LTM of the winning agent to see the merge
        winning_agent_ltm = agent_manager.agents[0].model.long_term_memory
        changed_weights = np.copy(winning_agent_ltm.layers[0].W)
        changed_weights += 0.5 # Introduce a noticeable change
        winning_agent_ltm.layers[0].W = changed_weights

        best_agents = agent_manager.collaborative_evaluation(
            evaluation_data=[([1, 2, 3], None)],
            tokenizer=tokenizer,
            top_k=1
        )
        agent_manager.merge_agents(best_agents)

        # 3. --- Assertions ---
        # The base model's LTM should now be updated
        updated_ltm_state = base_model.long_term_memory.get_state()
        updated_ltm_weights = updated_ltm_state['linear_0']['W']

        # Check that the winning agent was selected
        self.assertEqual(len(best_agents), 1)
        self.assertEqual(best_agents[0].agent_id, 'agent_0')

        # Check that the base model's LTM has changed from its initial state
        self.assertFalse(np.allclose(initial_ltm_weights, updated_ltm_weights),
                         "Base model's LTM weights did not change after merge.")

        # Check that the new LTM weights are the same as the winning agent's
        self.assertTrue(np.allclose(updated_ltm_weights, changed_weights),
                        "Merged LTM weights do not match the winning agent's weights.")

        print("Agent Evolution Cycle test PASSED.")

if __name__ == "__main__":
    unittest.main()
