"""
This module contains the TestAgent class, which tests the Agent's functionality.
"""
import unittest

from agent import Agent
from backend import np
from tests.test_utils import create_test_model


class TestAgent(unittest.TestCase):
    """
    Tests for the Agent class.
    """

    def setUp(self):
        """Set up the test environment."""
        self.model, self.config = create_test_model(ltm=True)
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
