"""
Unit tests for the AgentManager class.
"""
import unittest
from unittest.mock import MagicMock

import torch
from accelerate import Accelerator

from src.agent.agent_manager import AgentManager
from src.agent.evaluator import CollaborativeEvaluator
from src.config.core import TransformerConfig
from src.model.model import Transformer
from tests.test_utils import create_test_config, MockTokenizer


class TestAgentManager(unittest.TestCase):
    """Tests for the AgentManager."""

    def test_merge_agents_averages_ltm_weights(self):
        """Test that merge_agents correctly averages the LTM weights."""
        config = create_test_config()
        tokenizer = MockTokenizer()
        transformer_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            model=config.model,
            vision=config.vision,
            ltm=config.ltm,
            tokenizer=tokenizer,
        )
        model = Transformer(transformer_config)
        accelerator = Accelerator()
        model = accelerator.prepare(model)
        evaluator = CollaborativeEvaluator(agents=[], base_model=model)

        manager = AgentManager(
            model, num_agents=2, accelerator=accelerator, evaluator=evaluator
        )

        # Get the structure of the LTM state_dict from the base model
        unwrapped_model = accelerator.unwrap_model(model)
        ltm_state_structure = unwrapped_model.layers.long_term_memory.state_dict()

        # Create two mock states with this structure and known values
        with torch.no_grad():
            ltm_state_1 = {k: v.clone() for k, v in ltm_state_structure.items()}
            ltm_state_2 = {k: v.clone() for k, v in ltm_state_structure.items()}
            # Modify a specific weight tensor for the test
            # Get a real key from the state dict to avoid hardcoding
            key_to_test = next(iter(ltm_state_structure.keys()))
            ltm_state_1[key_to_test].fill_(1.0)
            ltm_state_2[key_to_test].fill_(3.0)

        # Mock the agents to return these controlled states
        manager.agents[0].get_ltm_state = MagicMock(return_value=ltm_state_1)
        manager.agents[1].get_ltm_state = MagicMock(return_value=ltm_state_2)

        manager.merge_agents(manager.agents)

        # Verify that the base model's LTM weights are the average
        unwrapped_model = accelerator.unwrap_model(model)
        merged_weight = unwrapped_model.layers.long_term_memory.state_dict()[key_to_test]
        self.assertTrue(torch.all(torch.eq(merged_weight, 2.0)))


if __name__ == '__main__':
    unittest.main()
