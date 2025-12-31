"""
Unit tests for the configuration models.
"""
import unittest

from pydantic import ValidationError

from src.config import ModelConfig


class TestConfigValidation(unittest.TestCase):
    """Tests for the configuration validation."""

    def test_valid_dynamic_architecture_config(self):
        """Tests that a valid dynamic architecture configuration passes validation."""
        try:
            ModelConfig(
                d_model=64,
                num_layers=12,
                num_heads=4,
                num_kv_heads=2,
                d_ff=128,
                max_seq_len=128,
                dropout_rate=0.1,
                early_exit_thresholds=[0.2, 0.6],
                early_exit_num_layers=[2, 6, 12],
                dynamic_moe_thresholds=[0.2, 0.6],
                dynamic_moe_k_values=[2, 4, 8],
            )
        except ValidationError as e:
            self.fail(f"Valid configuration failed validation with error: {e}")

    def test_invalid_early_exit_config_raises_error(self):
        """Tests that an invalid early exit configuration raises a ValueError."""
        with self.assertRaises(ValidationError) as context:
            ModelConfig(
                d_model=64,
                num_layers=12,
                num_heads=4,
                num_kv_heads=2,
                d_ff=128,
                max_seq_len=128,
                dropout_rate=0.1,
                early_exit_thresholds=[0.2, 0.6],
                early_exit_num_layers=[2, 6],  # Should have 3 values
            )
        self.assertIn(
            "Length of 'early_exit_num_layers' must be exactly one greater than",
            str(context.exception),
        )

    def test_invalid_dynamic_moe_config_raises_error(self):
        """Tests that an invalid dynamic MoE configuration raises a ValueError."""
        with self.assertRaises(ValidationError) as context:
            ModelConfig(
                d_model=64,
                num_layers=12,
                num_heads=4,
                num_kv_heads=2,
                d_ff=128,
                max_seq_len=128,
                dropout_rate=0.1,
                dynamic_moe_thresholds=[0.2, 0.6],
                dynamic_moe_k_values=[2, 4],  # Should have 3 values
            )
        self.assertIn(
            "Length of 'dynamic_moe_k_values' must be exactly one greater than",
            str(context.exception),
        )
