"""
Dataclasses for the agent module.
"""
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class LTMConfig:
    """Configuration for the agent's Long-Term Memory."""
    learning_rate: float
    surprise_threshold: float


@dataclass
class SpecializationConfig:
    """Configuration for the agent specialization process."""

    full_data: list
    seq_len: int
    batch_size: int
    steps_per_agent: int


@dataclass
class CollaborationContext:
    """Context for handling a collaboration request."""

    proposer: 'Agent'
    prompt_tokens: torch.Tensor
    scores: dict
    proposer_index: int
    response: torch.Tensor


@dataclass
class IndependentResponseContext:
    """Context for handling an independent response."""

    proposer: 'Agent'
    prompt_tokens: torch.Tensor
    scores: dict
    response: torch.Tensor


@dataclass
class CritiqueScoringContext:
    """Context for applying critique-based scores."""

    scores: dict
    avg_critique_score: float
    success_reward: float
    failure_penalty: float
    agents_to_reward: List['Agent']
    agents_to_penalize: List['Agent']
