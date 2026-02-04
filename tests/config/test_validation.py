"""
Tests for configuration validation using Pydantic.
"""
import unittest
from pydantic import ValidationError
from src.config.model_config import ModelConfig, VisionConfig, LTMArchitectureConfig

class TestConfigValidation(unittest.TestCase):
    """Verifies that invalid configurations raise Pydantic ValidationError."""

    def test_invalid_heads(self):
        """d_model must be divisible by num_heads."""
        with self.assertRaises(ValidationError):
            ModelConfig(
                vocab_size=1000,
                d_model=64,
                num_layers=2,
                num_heads=5, # 64 % 5 != 0
                num_kv_heads=1,
                d_ff=128,
                max_seq_len=512,
                dropout_rate=0.1
            )

    def test_invalid_kv_heads(self):
        """num_heads must be divisible by num_kv_heads."""
        with self.assertRaises(ValidationError):
            ModelConfig(
                vocab_size=1000,
                d_model=64,
                num_layers=2,
                num_heads=4,
                num_kv_heads=3, # 4 % 3 != 0
                d_ff=128,
                max_seq_len=512,
                dropout_rate=0.1
            )

    def test_invalid_moe_experts(self):
        """top_k_experts cannot exceed num_experts."""
        with self.assertRaises(ValidationError):
            ModelConfig(
                vocab_size=1000,
                d_model=64,
                num_layers=2,
                num_heads=4,
                num_kv_heads=2,
                d_ff=128,
                max_seq_len=512,
                dropout_rate=0.1,
                num_experts=4,
                top_k_experts=5 # 5 > 4
            )

    def test_invalid_vision_patches(self):
        """image_size must be divisible by patch_size."""
        with self.assertRaises(ValidationError):
            VisionConfig(
                image_size=(224, 224),
                patch_size=17 # 224 % 17 != 0
            )

if __name__ == "__main__":
    unittest.main()
