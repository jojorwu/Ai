"""
Unit tests for the Transformer model.
"""
import unittest
from unittest.mock import MagicMock, patch

import torch

from src.config.core import (TrainConfig, TransformerConfig, ModelConfig,
                             VisionConfig, LTMArchitectureConfig)
from src.model.model import Transformer


class TestTransformer(unittest.TestCase):
    """Tests for the Transformer model."""

    def setUp(self):
        """Set up a mock model and config for testing."""
        config = TrainConfig.from_json('config_train.json')
        config.model.vocab_size = 50
        self.config = TransformerConfig(
            vocab_size=config.model.vocab_size,
            model=config.model,
            vision=config.vision,
            ltm=config.ltm,
            tokenizer=None,
        )
        self.model = Transformer(self.config)

    def test_validate_and_accept_chunk_all_match(self):
        """
        Tests the _validate_and_accept_chunk method when all tokens in the
        speculative chunk match the verification tokens.
        """
        true_logits = torch.randn(1, 5, self.config.vocab_size)
        speculative_chunk = torch.randint(0, self.config.vocab_size, (1, 5))
        inputs = MagicMock()
        inputs.sampling_config.temperature = 0.0 # Force greedy sampling
        inputs.sampling_config.top_k = 0
        inputs.sampling_config.dynamic_top_k = None

        with patch.object(self.model, '_sample_from_logits', return_value=speculative_chunk):
            accepted_chunk = self.model._validate_and_accept_chunk(  # pylint: disable=protected-access
                true_logits, speculative_chunk, inputs
            )
            self.assertTrue(torch.equal(accepted_chunk, speculative_chunk))

    def test_validate_and_accept_chunk_partial_match(self):
        """
        Tests the _validate_and_accept_chunk method when some tokens in the
        speculative chunk match the verification tokens.
        """
        true_logits = torch.randn(1, 5, self.config.vocab_size)
        speculative_chunk = torch.randint(0, self.config.vocab_size, (1, 5))
        verification_tokens = speculative_chunk.clone()
        verification_tokens[0, 3] = (verification_tokens[0, 3] + 1) % self.config.vocab_size
        inputs = MagicMock()
        inputs.sampling_config.temperature = 0.0 # Force greedy sampling
        inputs.sampling_config.top_k = 0
        inputs.sampling_config.dynamic_top_k = None

        with patch.object(self.model, '_sample_from_logits', return_value=verification_tokens):
            accepted_chunk = self.model._validate_and_accept_chunk(  # pylint: disable=protected-access
                true_logits, speculative_chunk, inputs
            )
            expected_chunk = torch.cat(
                [
                    speculative_chunk[:, :3],
                    verification_tokens[:, 3].unsqueeze(-1)
                ],
                dim=1
            )
            self.assertTrue(torch.equal(accepted_chunk, expected_chunk))

    def test_validate_and_accept_chunk_no_match(self):
        """
        Tests the _validate_and_accept_chunk method when no tokens in the
        speculative chunk match the verification tokens.
        """
        true_logits = torch.randn(1, 5, self.config.vocab_size)
        speculative_chunk = torch.randint(0, self.config.vocab_size, (1, 5))
        verification_tokens = (speculative_chunk + 1) % self.config.vocab_size
        inputs = MagicMock()
        inputs.sampling_config.temperature = 0.0 # Force greedy sampling
        inputs.sampling_config.top_k = 0
        inputs.sampling_config.dynamic_top_k = None

        with patch.object(self.model, '_sample_from_logits', return_value=verification_tokens):
            accepted_chunk = self.model._validate_and_accept_chunk(  # pylint: disable=protected-access
                true_logits, speculative_chunk, inputs
            )
            expected_chunk = verification_tokens[:, 0].unsqueeze(-1)
            self.assertTrue(torch.equal(accepted_chunk, expected_chunk))

    def test_layer_skipping(self):
        """
        Tests that the model correctly skips layers by mocking the GatingNetwork.
        """
        with patch.object(
            self.model.layers.gating_network,
            'forward',
            return_value=(torch.tensor([3]), torch.tensor([1])) # Mock: use 3 layers, 1 expert
        ) as mock_gate:
            with patch(
                'src.model.model.DecoderBlock.forward',
                return_value=(torch.randn(1, 10, 64), None)
            ) as mock_decoder:
                self.model(torch.randint(0, self.config.vocab_size, (1, 10)))
                mock_gate.assert_called_once()
                self.assertEqual(mock_decoder.call_count, 3)

    def test_dynamic_expert_allocation(self):
        """
        Tests that the model correctly allocates experts by mocking the GatingNetwork.
        """
        with patch.object(
            self.model.layers.gating_network,
            'forward',
            return_value=(torch.tensor([4]), torch.tensor([5])) # Mock: use 4 layers, 5 experts
        ) as mock_gate:
            with patch(
                'src.model.model.DecoderBlock.forward',
                return_value=(torch.randn(1, 10, 64), None)
            ) as mock_decoder:
                self.model(torch.randint(0, self.config.vocab_size, (1, 10)))
                mock_gate.assert_called_once()
                # The 'dynamic_top_k' is passed within the ForwardPassInput dataclass
                final_call_args = mock_decoder.call_args[0][0]
                self.assertEqual(final_call_args.dynamic_top_k, 5)

    def test_top_p_sampling(self):
        """Tests the top-p (nucleus) sampling logic."""
        # Test case 1: One token is overwhelmingly likely
        logits1 = torch.tensor([[0.1, 0.2, 0.3, 0.4, 10.0]])
        top_p1 = 0.9
        next_token1 = self.model._sample_from_logits(  # pylint: disable=protected-access
            logits1, temperature=1.0, top_k=0, top_p=top_p1
        )
        self.assertEqual(next_token1.item(), 4)

        # Test case 2: More evenly distributed probabilities
        logits2 = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        top_p2 = 0.4  # This should select tokens 4 and 3

        tokens = [
            self.model._sample_from_logits(  # pylint: disable=protected-access
                logits2, temperature=1.0, top_k=0, top_p=top_p2
            ).item() for _ in range(100)
        ]

        # Check that the sampled tokens are only from the top-p nucleus
        self.assertTrue(all(t in [3, 4] for t in tokens))
        # Check that we have some variety, confirming it's not just greedy sampling
        self.assertTrue(len(set(tokens)) > 1)
