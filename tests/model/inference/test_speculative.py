"""
Unit tests for the SpeculativeEngine.
"""
import unittest
from unittest.mock import MagicMock, patch
import torch
from src.model.inference.speculative import SpeculativeEngine

class TestSpeculativeEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SpeculativeEngine()

    def test_validate_chunk_all_match(self):
        true_logits = torch.randn(1, 5, 50)
        speculative_chunk = torch.randint(0, 50, (1, 5))
        sampling_config = MagicMock()
        sampling_config.temperature = 0.0
        sampling_config.top_k = 0
        sampling_config.top_p = 0.0
        sampling_config.dynamic_top_k = None

        with patch.object(self.engine.sampler, 'sample', return_value=speculative_chunk):
            accepted_chunk = self.engine.validate_chunk(
                true_logits, speculative_chunk, sampling_config
            )
            self.assertTrue(torch.equal(accepted_chunk, speculative_chunk))

    def test_validate_chunk_partial_match(self):
        true_logits = torch.randn(1, 5, 50)
        speculative_chunk = torch.randint(0, 50, (1, 5))
        verification_tokens = speculative_chunk.clone()
        verification_tokens[0, 3] = (verification_tokens[0, 3] + 1) % 50
        sampling_config = MagicMock()
        sampling_config.temperature = 0.0
        sampling_config.top_k = 0
        sampling_config.top_p = 0.0
        sampling_config.dynamic_top_k = None

        with patch.object(self.engine.sampler, 'sample', return_value=verification_tokens):
            accepted_chunk = self.engine.validate_chunk(
                true_logits, speculative_chunk, sampling_config
            )
            expected_chunk = torch.cat(
                [
                    speculative_chunk[:, :3],
                    verification_tokens[:, 3].unsqueeze(-1)
                ],
                dim=1
            )
            self.assertTrue(torch.equal(accepted_chunk, expected_chunk))

    def test_validate_chunk_no_match(self):
        true_logits = torch.randn(1, 5, 50)
        speculative_chunk = torch.randint(0, 50, (1, 5))
        verification_tokens = (speculative_chunk + 1) % 50
        sampling_config = MagicMock()
        sampling_config.temperature = 0.0
        sampling_config.top_k = 0
        sampling_config.top_p = 0.0
        sampling_config.dynamic_top_k = None

        with patch.object(self.engine.sampler, 'sample', return_value=verification_tokens):
            accepted_chunk = self.engine.validate_chunk(
                true_logits, speculative_chunk, sampling_config
            )
            expected_chunk = verification_tokens[:, 0].unsqueeze(-1)
            self.assertTrue(torch.equal(accepted_chunk, expected_chunk))

if __name__ == "__main__":
    unittest.main()
