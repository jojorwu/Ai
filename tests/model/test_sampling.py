"""
Unit tests for the LogitSampler.
"""
import unittest
import torch
from src.model.sampling import LogitSampler

class TestLogitSampler(unittest.TestCase):
    def test_top_p_sampling(self):
        """Tests the top-p (nucleus) sampling logic."""
        sampler = LogitSampler()
        # Test case 1: One token is overwhelmingly likely
        logits1 = torch.tensor([[0.1, 0.2, 0.3, 0.4, 10.0]])
        top_p1 = 0.9
        next_token1 = sampler.sample(
            logits1, temperature=1.0, top_k=0, top_p=top_p1
        )
        self.assertEqual(next_token1.item(), 4)

        # Test case 2: More evenly distributed probabilities
        logits2 = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        top_p2 = 0.4  # This should select tokens 4 and 3

        tokens = [
            sampler.sample(
                logits2, temperature=1.0, top_k=0, top_p=top_p2
            ).item() for _ in range(100)
        ]

        # Check that the sampled tokens are only from the top-p nucleus
        self.assertTrue(all(t in [3, 4] for t in tokens))
        # Check that we have some variety
        self.assertTrue(len(set(tokens)) > 1)

if __name__ == "__main__":
    unittest.main()
