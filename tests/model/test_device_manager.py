"""
Unit tests for the DeviceManager.
"""
import unittest
from unittest.mock import patch, MagicMock
import torch
from src.model.device_manager import DeviceManager
from src.config.hardware_config import HardwareConfig


class TestDeviceManager(unittest.TestCase):
    """Tests for the DeviceManager class."""

    def test_optimize_environment_cpu(self):
        """Tests that CPU optimizations are applied correctly."""
        config = HardwareConfig(
            device="cpu",
            num_threads=2,
            num_interop_threads=1,
            enable_mkldnn=True,
            flush_denormals=True
        )
        manager = DeviceManager(config)

        with patch("torch.set_num_threads") as mock_set_threads, \
             patch("torch.set_num_interop_threads") as mock_set_interop, \
             patch("torch.set_flush_denormal") as mock_set_flush:

            manager.optimize_environment()

            mock_set_threads.assert_called_once_with(2)
            mock_set_interop.assert_called_once_with(1)
            mock_set_flush.assert_called_once_with(True)
            self.assertTrue(torch.backends.mkldnn.enabled)

    def test_optimize_environment_gpu(self):
        """Tests that GPU optimizations are applied correctly."""
        config = HardwareConfig(device="gpu")
        # Mock cuda available
        with patch("torch.cuda.is_available", return_value=True):
            manager = DeviceManager(config)

            manager.optimize_environment()

            self.assertTrue(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)

    def test_get_device_map_hybrid(self):
        """Tests the hybrid device mapping strategy."""
        config = HardwareConfig(strategy="hybrid")
        with patch("torch.cuda.is_available", return_value=True):
            manager = DeviceManager(config)
            device_map = manager.get_device_map()
            self.assertEqual(device_map["layers.long_term_memory"], "cpu")
            self.assertEqual(device_map[""], "cuda:0")


if __name__ == "__main__":
    unittest.main()
