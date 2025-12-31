"""
Unit tests for the DeviceManager.
"""
import unittest
from unittest.mock import patch

from src.config import HardwareConfig
from src.device_manager import DeviceManager


class TestDeviceManager(unittest.TestCase):
    """Tests for the DeviceManager."""

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
        """Tests that the hybrid strategy falls back to 'auto' without CUDA."""
        config = HardwareConfig(strategy="hybrid")
        manager = DeviceManager(config)
        device_map = manager.get_device_map()
        self.assertEqual(device_map, "auto")
        self.assertTrue(manager.should_disable_4bit())

    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_only_strategy(self, _mock_cuda_available):
        """Tests that a non-hybrid strategy returns 'auto'."""
        config = HardwareConfig(strategy="discrete")
        manager = DeviceManager(config)
        device_map = manager.get_device_map()
        self.assertEqual(device_map, "auto")
        self.assertTrue(manager.should_disable_4bit())
