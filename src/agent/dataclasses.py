"""
Dataclasses for the agent module.
"""
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.model.complexity_manager import ComplexityManager
    from src.model.layers.kv_cache import KVCache


@dataclass
class AgentState:
    """Keeps track of the agent's state during a conversation."""

    conversation_history_tokens: List[int]
    complexity_manager: 'ComplexityManager' = None
    main_cache: 'KVCache' = None
    draft_cache: 'KVCache' = None

    def get_new_tokens(self) -> List[int]:
        """Returns the tokens that have not yet been processed by the KV cache."""
        if self.main_cache is None:
            return self.conversation_history_tokens
        cached_len = self.main_cache.current_pos
        return self.conversation_history_tokens[cached_len:]

    def append_tokens(self, tokens: List[int]):
        """Appends new tokens to the conversation history."""
        self.conversation_history_tokens.extend(tokens)


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
