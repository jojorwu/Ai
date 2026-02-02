"""
Tests for the VisionEncoder layer.
"""
import unittest
import torch
from src.model.layers.core.vision import VisionEncoder
from src.config.model_config import VisionConfig

class TestVisionEncoder(unittest.TestCase):
    def test_vision_encoder_output_shape(self):
        d_model = 128
        config = VisionConfig(
            image_size=(224, 224),
            patch_size=16,
            num_channels=3
        )
        encoder = VisionEncoder(config, d_model)

        # Batch size 2, 3 channels, 224x224 image
        images = torch.randn(2, 3, 224, 224)
        output = encoder(images)

        # num_patches = (224/16) * (224/16) = 14 * 14 = 196
        expected_shape = (2, 196, d_model)
        self.assertEqual(output.shape, expected_shape)

    def test_vision_encoder_invalid_image_size(self):
        d_model = 128
        config = VisionConfig(
            image_size=(224, 224),
            patch_size=16,
            num_channels=3
        )
        encoder = VisionEncoder(config, d_model)

        # Image size doesn't match config (should still work if divisible by patch_size,
        # but output patches will differ from pos_embed if not handled)
        # Actually in my implementation it just uses image_size for num_patches in __init__
        images = torch.randn(2, 3, 128, 128)

        with self.assertRaises(RuntimeError):
            # This will fail at "x = x + self.pos_embed" because shapes won't match
            # 128/16 = 8. 8*8 = 64 patches. self.pos_embed has 196.
            encoder(images)

if __name__ == '__main__':
    unittest.main()
