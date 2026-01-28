"""
Defines update policies for agent training.
"""
from abc import ABC, abstractmethod
import torch
from src.agent.dataclasses import LTMConfig


class UpdatePolicy(ABC):
    """
    Base class for LTM update policies.
    """

    @abstractmethod
    def should_update(self, surprise: float, ltm_config: LTMConfig) -> bool:
        """Determines if an update should be performed based on surprise."""


class SurpriseUpdatePolicy(UpdatePolicy):
    """
    Updates LTM if surprise exceeds a fixed threshold.
    """

    def should_update(self, surprise: float, ltm_config: LTMConfig) -> bool:
        return surprise > ltm_config.surprise_threshold
