"""
This module contains the ComplexityManager class, which dynamically adjusts
generation parameters based on model 'surprise'.
"""
from dataclasses import dataclass


@dataclass
class DynamicParametersConfig:
    """Configuration for dynamic parameter adjustment."""
    low_complexity_threshold: float
    medium_complexity_threshold: float
    high_complexity_threshold: float
    low_complexity_top_k: int
    medium_complexity_top_k: int
    high_complexity_top_k: int

class ComplexityManager:
    """Manages the dynamic complexity level based on a moving average of 'surprise'."""

    def __init__(self, config: DynamicParametersConfig, window_size: int = 5):
        self.config = config
        self.surprise_history = []
        self.window_size = window_size
        self.current_complexity = "low"

    def update_surprise(self, surprise_value: float):
        """Adds a new surprise value and updates the complexity level."""
        self.surprise_history.append(surprise_value)
        if len(self.surprise_history) > self.window_size:
            self.surprise_history.pop(0)
        self._update_complexity()

    def _update_complexity(self):
        """Determines the complexity level based on the average surprise."""
        if not self.surprise_history:
            self.current_complexity = "low"
            return

        avg_surprise = sum(self.surprise_history) / len(self.surprise_history)

        if avg_surprise >= self.config.high_complexity_threshold:
            self.current_complexity = "high"
        elif avg_surprise >= self.config.medium_complexity_threshold:
            self.current_complexity = "medium"
        else:
            self.current_complexity = "low"

    def get_top_k(self) -> int:
        """Returns the top_k value for the current complexity level."""
        if self.current_complexity == "high":
            return self.config.high_complexity_top_k
        if self.current_complexity == "medium":
            return self.config.medium_complexity_top_k
        return self.config.low_complexity_top_k
