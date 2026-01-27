"""
PyTorch implementation of the Agent class for the evolutionary training approach.
"""
import copy
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.optim import Adam

from src.agent.dataclasses import LTMConfig
from src.model.model import Transformer
from src.model.structures import GenerateInput, SamplingConfig

if TYPE_CHECKING:
    from src.model.layers.kv_cache import KVCache


@dataclass
class AgentMetrics:
    """Keeps track of an agent's performance metrics."""
    total_surprise: float = 0.0
    experience_count: int = 0
    value_score_sum: float = 0.0
    fitness_score: float = field(default=-float('inf'))

class Agent:
    """
    Represents a single "agent" with its own long-term memory (LTM), adapted for PyTorch.
    The agent shares the base model's weights but has a unique LTM.
    """

    def __init__(
        self,
        base_model: Transformer,
        ltm_config: LTMConfig,
        agent_id: str | None = None,
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.base_model = base_model
        self.ltm_config = ltm_config
        self.long_term_memory = (
            copy.deepcopy(base_model.layers.long_term_memory)
            if base_model.layers.long_term_memory
            else None
        )

        if self.long_term_memory:
            self.ltm_optimizer = Adam(
                self.long_term_memory.parameters(),
                lr=self.ltm_config.learning_rate,
            )
        else:
            self.ltm_optimizer = None

        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()  # Used to push value towards 1.0
        self.metrics = AgentMetrics()

    def get_ltm_state(self) -> dict | None:
        """Returns the state_dict of this agent's LTM."""
        return self.long_term_memory.state_dict() if self.long_term_memory else None

    def get_fitness_score(self) -> float:
        """Calculates the agent's fitness."""
        return self.metrics.fitness_score

    def update_fitness_score(self, score: float):
        """Updates the agent's fitness score."""
        self.metrics.fitness_score = score

    @torch.no_grad()
    def generate_response(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 50,
        kv_cache: 'KVCache' = None,
        draft_cache: 'KVCache' = None,
    ) -> torch.Tensor:
        """
        Generates a full response tensor based on a prompt, utilizing optional KV caches.
        """
        self.base_model.eval()
        if self.long_term_memory:
            self.long_term_memory.eval()

        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)

        sampling_config = SamplingConfig(temperature=0.7, top_k=50)
        generate_input = GenerateInput(
            start_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            sampling_config=sampling_config,
            ltm_override=self.long_term_memory,
            kv_cache=kv_cache,
            draft_cache=draft_cache,
        )

        # Accumulate all chunks from the generator
        full_response = prompt_tokens
        for chunk, _ in self.base_model.generate(generate_input):
            full_response = torch.cat([full_response, chunk], dim=1)

        return full_response
