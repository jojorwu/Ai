"""
Tests for thread-safety and parallel execution in agent training and evaluation.
"""
import unittest
from concurrent.futures import ThreadPoolExecutor
import torch
from torch import nn
from src.agent.agent import Agent
from src.agent.trainer import AgentTrainer
from src.agent.dataclasses import LTMConfig
from src.model.model import Transformer
from tests.test_utils import create_test_config


class TestThreadSafety(unittest.TestCase):
    """Verifies that parallel agent processing is thread-safe."""

    def setUp(self):
        self.config = create_test_config()
        # Mock base model with parameters that require grad
        self.base_model = Transformer(self.config.to_transformer_config())
        for p in self.base_model.parameters():
            p.requires_grad = True

        self.ltm_config = LTMConfig(
            learning_rate=0.01,
            surprise_threshold=0.0
        )

    def test_isolated_gradients_in_trainer(self):
        """
        Ensures that training one agent doesn't affect the base model's gradients
        or another agent's gradients.
        """
        agent1 = Agent(self.base_model, self.ltm_config, agent_id="agent1")
        agent2 = Agent(self.base_model, self.ltm_config, agent_id="agent2")

        trainer1 = AgentTrainer(agent1)
        trainer2 = AgentTrainer(agent2)

        vocab_size = self.base_model.config.vocab_size
        x = torch.randint(0, vocab_size, (1, 8))
        y = torch.randint(0, vocab_size, (1, 8))

        # Zero out all gradients initially
        self.base_model.zero_grad()
        for p in agent1.long_term_memory.parameters():
            p.grad = None
        for p in agent2.long_term_memory.parameters():
            p.grad = None

        # Train agent 1
        trainer1.experience(x, y)

        # Check that agent 1 has gradients on its LTM
        has_grads1 = any(p.grad is not None for p in agent1.long_term_memory.parameters())
        self.assertTrue(has_grads1)

        # Check that base model still has NO gradients
        for name, p in self.base_model.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.all(p.grad == 0), f"Parameter {name} has non-zero gradient")

        # Check that agent 2 has NO gradients
        for p in agent2.long_term_memory.parameters():
            self.assertIsNone(p.grad)

    def test_parallel_specialization_throughput(self):
        """
        Briefly checks if parallel specialization runs without crashing.
        """
        from src.agent.specializer import AgentSpecializer
        from src.agent.dataclasses import SpecializationConfig

        agents = [Agent(self.base_model, self.ltm_config, agent_id=f"a{i}") for i in range(4)]
        specializer = AgentSpecializer()

        vocab_size = self.base_model.config.vocab_size
        spec_config = SpecializationConfig(
            full_data=list(range(vocab_size)) * 4,
            seq_len=4,
            batch_size=2,
            steps_per_agent=2
        )

        # This should run in parallel using ThreadPoolExecutor internally
        specializer.specialize_agents(agents, spec_config, torch.device("cpu"))

        # Verify all agents gained some experience
        for agent in agents:
            self.assertGreater(agent.metrics.experience_count, 0)


if __name__ == "__main__":
    unittest.main()
