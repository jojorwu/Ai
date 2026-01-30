"""
Handles KV cache initialization and management for agents.
"""
import torch
from accelerate import Accelerator
from src.agent.dataclasses import AgentState
from src.model.model import Transformer
from src.model.layers.attention.kv_cache import KVCache, KVCacheConfig


class CacheManager:
    """
    Manages the lifecycle of KV caches for agents.
    """

    def __init__(self, accelerator: Accelerator):
        self.accelerator = accelerator

    def initialize_agent_caches(self, agent_state: AgentState, model: Transformer):
        """Initializes KV caches for the main and draft models."""
        unwrapped_model = self.accelerator.unwrap_model(model)
        d_k = unwrapped_model.config.model.d_model // unwrapped_model.config.model.num_heads

        agent_state.main_cache = KVCache(
            KVCacheConfig(
                num_layers=unwrapped_model.config.model.num_layers,
                batch_size=1,
                num_kv_heads=unwrapped_model.config.model.num_kv_heads,
                d_k=d_k,
                max_seq_len=unwrapped_model.config.model.max_seq_len,
                anchor_size=unwrapped_model.config.model.anchor_window_size,
            ),
            device=self.accelerator.device,
            dtype=unwrapped_model.layers.embedding.weight.dtype,
        )

        if (
            unwrapped_model.draft_model
            and unwrapped_model.draft_model is not unwrapped_model
        ):
            agent_state.draft_cache = KVCache(
                KVCacheConfig(
                    num_layers=unwrapped_model.draft_model.config.model.num_layers,
                    batch_size=1,
                    num_kv_heads=unwrapped_model.draft_model.config.model.num_kv_heads,
                    d_k=unwrapped_model.draft_model.config.model.d_model
                    // unwrapped_model.draft_model.config.model.num_heads,
                    max_seq_len=unwrapped_model.draft_model.config.model.max_seq_len,
                    anchor_size=unwrapped_model.draft_model.config.model.anchor_window_size,
                ),
                device=self.accelerator.device,
                dtype=unwrapped_model.layers.embedding.weight.dtype,
            )
        else:
            agent_state.draft_cache = agent_state.main_cache
