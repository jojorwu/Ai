"""
Unit tests for the DeviceManager.
"""
import unittest
from unittest.mock import patch
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

    @patch("torch.cuda.is_available", return_value=True)
    def test_hybrid_strategy_with_cuda(self, _mock_cuda_available):
        """Tests that the hybrid strategy returns the correct device map with CUDA."""
        config = HardwareConfig(strategy="hybrid")
        manager = DeviceManager(config)
        device_map = manager.get_device_map()
        expected_map = {"layers.long_term_memory": "cpu", "": "cuda:0"}
        self.assertEqual(device_map, expected_map)
        self.assertFalse(manager.should_disable_4bit())

    @patch("torch.cuda.is_available", return_value=False)
    def test_hybrid_strategy_without_cuda(self, _mock_cuda_available):
        """Tests that the hybrid strategy falls back to the default device without CUDA."""
        config = HardwareConfig(strategy="hybrid", device="cpu")
        manager = DeviceManager(config)
        device_map = manager.get_device_map()
        self.assertEqual(device_map, {"": "cpu"})
        self.assertTrue(manager.should_disable_4bit())

    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_only_strategy(self, _mock_cuda_available):
        """Tests that a non-hybrid strategy returns the default device."""
        config = HardwareConfig(strategy="discrete", device="cpu")
        manager = DeviceManager(config)
        device_map = manager.get_device_map()
        self.assertEqual(device_map, {"": "cpu"})
        self.assertTrue(manager.should_disable_4bit())


if __name__ == "__main__":
    unittest.main()
