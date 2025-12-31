"""
Unit tests for the Transformer model.
"""
import unittest
from unittest.mock import MagicMock, patch

import torch

from src.config import TransformerConfig, ModelConfig, VisionConfig, LTMArchitectureConfig
from src.model import Transformer


class TestTransformer(unittest.TestCase):
    """Tests for the Transformer model."""

    def setUp(self):
        """Set up a mock model and config for testing."""
        self.config = TransformerConfig(
            vocab_size=100,
            model=ModelConfig(
                d_model=64,
                num_layers=12,
                num_heads=4,
                num_kv_heads=2,
                d_ff=128,
                max_seq_len=128,
                dropout_rate=0.1,
                ltm=LTMArchitectureConfig(d_hidden=32, num_layers=1),
                early_exit_thresholds=[0.2, 0.6],
                early_exit_num_layers=[2, 6, 12],
                dynamic_moe_thresholds=[0.2, 0.6],
                dynamic_moe_k_values=[2, 4, 8],
            ),
            vision=VisionConfig(),
            ltm=None,
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
            accepted_chunk = self.model._validate_and_accept_chunk( # pylint: disable=protected-access
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
            accepted_chunk = self.model._validate_and_accept_chunk( # pylint: disable=protected-access
                true_logits, speculative_chunk, inputs
            )
            expected_chunk = torch.cat(
                [speculative_chunk[:, :3], verification_tokens[:, 3].unsqueeze(-1)], dim=1
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
            accepted_chunk = self.model._validate_and_accept_chunk( # pylint: disable=protected-access
                true_logits, speculative_chunk, inputs
            )
            expected_chunk = verification_tokens[:, 0].unsqueeze(-1)
            self.assertTrue(torch.equal(accepted_chunk, expected_chunk))

    def test_layer_skipping(self):
        """
        Tests that the model correctly skips layers based on the complexity score.
        """
        # Low complexity
        ltm_low_complexity = (torch.randn(1, 1, 64), torch.tensor([[[0.1]]]))
        with patch.object(self.model.layers.long_term_memory, 'forward', return_value=ltm_low_complexity):
            with patch('src.model.DecoderBlock.forward', return_value=(torch.randn(1, 10, 64), None)) as mock:
                self.model(torch.randint(0, self.config.vocab_size, (1, 10)))
                self.assertEqual(mock.call_count, 2)

        # Medium complexity
        ltm_med_complexity = (torch.randn(1, 1, 64), torch.tensor([[[0.4]]]))
        with patch.object(self.model.layers.long_term_memory, 'forward', return_value=ltm_med_complexity):
            with patch('src.model.DecoderBlock.forward', return_value=(torch.randn(1, 10, 64), None)) as mock:
                self.model(torch.randint(0, self.config.vocab_size, (1, 10)))
                self.assertEqual(mock.call_count, 6)

        # High complexity
        ltm_high_complexity = (torch.randn(1, 1, 64), torch.tensor([[[0.8]]]))
        with patch.object(self.model.layers.long_term_memory, 'forward', return_value=ltm_high_complexity):
            with patch('src.model.DecoderBlock.forward', return_value=(torch.randn(1, 10, 64), None)) as mock:
                self.model(torch.randint(0, self.config.vocab_size, (1, 10)))
                self.assertEqual(mock.call_count, 12)

    def test_dynamic_expert_allocation(self):
        """
        Tests that the model correctly allocates experts based on the complexity score.
        """
        # Low complexity
        ltm_low_complexity = (torch.randn(1, 1, 64), torch.tensor([[[0.1]]]))
        with patch.object(self.model.layers.long_term_memory, 'forward', return_value=ltm_low_complexity):
            with patch('src.model.DecoderBlock.forward', return_value=(torch.randn(1, 10, 64), None)) as mock:
                self.model(torch.randint(0, self.config.vocab_size, (1, 10)))
                self.assertEqual(mock.call_args[0][0].dynamic_top_k, 2)

        # Medium complexity
        ltm_med_complexity = (torch.randn(1, 1, 64), torch.tensor([[[0.4]]]))
        with patch.object(self.model.layers.long_term_memory, 'forward', return_value=ltm_med_complexity):
            with patch('src.model.DecoderBlock.forward', return_value=(torch.randn(1, 10, 64), None)) as mock:
                self.model(torch.randint(0, self.config.vocab_size, (1, 10)))
                self.assertEqual(mock.call_args[0][0].dynamic_top_k, 4)

        # High complexity
        ltm_high_complexity = (torch.randn(1, 1, 64), torch.tensor([[[0.8]]]))
        with patch.object(self.model.layers.long_term_memory, 'forward', return_value=ltm_high_complexity):
            with patch('src.model.DecoderBlock.forward', return_value=(torch.randn(1, 10, 64), None)) as mock:
                self.model(torch.randint(0, self.config.vocab_size, (1, 10)))
                self.assertEqual(mock.call_args[0][0].dynamic_top_k, 8)

    def test_forward_pass_handles_invalid_complexity_score(self):
        """
        Tests that the forward pass handles invalid complexity scores gracefully.
        """
        invalid_scores = [
            None,
            torch.tensor([]),
            torch.tensor([float('nan')]),
            torch.tensor([float('inf')])
        ]

        for score in invalid_scores:
            with self.subTest(score=score):
                ltm_output = (torch.randn(1, 1, 64), score)
                with patch.object(self.model.layers.long_term_memory, 'forward', return_value=ltm_output):
                    with patch('src.model.DecoderBlock.forward', return_value=(torch.randn(1, 10, 64), None)) as mock:
                        self.model(torch.randint(0, self.config.vocab_size, (1, 10)))

                        min_layers = self.config.model.early_exit_num_layers[0]
                        self.assertEqual(mock.call_count, min_layers)

                        min_experts = self.config.model.dynamic_moe_k_values[0]
                        self.assertEqual(mock.call_args[0][0].dynamic_top_k, min_experts)
