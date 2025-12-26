"""
This module contains the TestAgent class, which tests the Agent's functionality.
"""
import unittest

from backend import np
from agent import Agent
from config import Config
from model import Transformer


class TestAgent(unittest.TestCase):
    """
    Tests for the Agent class.
    """

    def setUp(self):
        """Set up the test environment."""
        self.config = Config.from_json('config.json')
        # Ensure LTM is enabled for this test by setting its dimensions
        self.config.model.ltm_d_hidden = 64
        self.config.model.ltm_num_layers = 2
        self.model = Transformer(
            vocab_size=50,
            model_config=self.config.model,
            vision_config=self.config.vision,
            ltm_config=self.config.ltm
        )
        self.agent = Agent(self.model)

    def test_agent_experience(self):
        """
        Tests that the agent's experience method runs without errors.
        """
        x_batch = np.random.randint(0, 50, (2, 10))
        y_batch = np.random.randint(0, 50, (2, 10))
        image_batch = np.random.rand(2, 10, 224, 224, 3).astype(np.float32)
        self.agent.experience(x_batch, y_batch, image_batch)
        self.assertEqual(self.agent.metrics.experience_count, 1)


if __name__ == '__main__':
    unittest.main()
