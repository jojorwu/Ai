"""
Dataclasses for the agent module.
"""
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.model.titans.complexity_manager import ComplexityManager
    from src.model.layers.attention.kv_cache import KVCache


@dataclass
class AgentState:
    """Keeps track of the agent's state during a conversation."""

    conversation_history_tokens: List[int]
    complexity_manager: 'ComplexityManager' = None
    main_cache: 'KVCache' = None
    draft_cache: 'KVCache' = None
    # Persistent memory matrix for the Linear Associative LTM
    ltm_memory: torch.Tensor = None
    # Number of tokens from the current conversation_history_tokens that have been processed.
    _processed_count: int = 0

    def get_new_tokens(self) -> List[int]:
        """Returns the tokens that have not yet been processed by the KV cache."""
        return self.conversation_history_tokens[self._processed_count:]

    def mark_as_processed(self, count: int):
        """Marks the given number of tokens as processed."""
        self._processed_count = min(len(self.conversation_history_tokens), self._processed_count + count)

    def append_tokens(self, tokens: List[int]):
        """Appends new tokens to the conversation history."""
        self.conversation_history_tokens.extend(tokens)

    def prune_history(self, context_window_size: int):
        """
        Prunes the conversation history to fit within the context window
        using the "Attention Sink" principle (Anchors + Sliding Window).
        This optimizes processor usage by allowing the KV cache to remain persistent.
        """
        if len(self.conversation_history_tokens) <= context_window_size:
            return

        # Determine anchor size from cache if available, else default to 4
        anchor_size = 4
        if self.main_cache:
            anchor_size = self.main_cache.config.anchor_size

        # Keep anchors and the most recent tokens
        anchors = self.conversation_history_tokens[:anchor_size]
        recent_tokens_count = context_window_size - anchor_size
        recent = self.conversation_history_tokens[-recent_tokens_count:]

        self.conversation_history_tokens = anchors + recent

        # After pruning, all tokens in the new history are already in the cache.
        self._processed_count = len(self.conversation_history_tokens)

        # IMPORTANT: We do NOT reset the KV cache positions here.
        # The Anchor-Aware KVCache handles the sliding window logically.
        # This prevents expensive re-encoding of the prompt on the CPU/GPU.


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
