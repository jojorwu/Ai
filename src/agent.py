"""
PyTorch implementation of the Agent class for the evolutionary training approach.
"""
import copy
import uuid
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.optim import Adam

from model import GenerateInput, SamplingConfig, Transformer


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

    def __init__(self, base_model: Transformer, agent_id: str | None = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.base_model = base_model
        self.long_term_memory = (
            copy.deepcopy(base_model.layers.long_term_memory)
            if base_model.layers.long_term_memory
            else None
        )

        if self.long_term_memory:
            self.ltm_optimizer = Adam(
                self.long_term_memory.parameters(),
                lr=self.base_model.config.ltm.optimizer.learning_rate,
            )
        else:
            self.ltm_optimizer = None

        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()  # Used to push value towards 1.0
        self.metrics = AgentMetrics()

    def _update_ltm_and_calc_surprise(self) -> float:
        """
        Calculates the gradient norm for LTM parameters ("surprise") and,
        if it exceeds a threshold, performs an optimizer step.
        """
        if not self.long_term_memory:
            return 0.0

        grad_tensors = [
            p.grad.detach().flatten()
            for p in self.long_term_memory.parameters()
            if p.grad is not None
        ]

        if not grad_tensors:
            return 0.0

        surprise = torch.norm(torch.cat(grad_tensors)).item()
        self.metrics.total_surprise += surprise

        if surprise > self.base_model.config.ltm.surprise_threshold:
            self.ltm_optimizer.step()

        return surprise

    def experience(self, x_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        The process of an agent gaining "experience" in a batch training mode.
        This method updates the agent's LTM based on the surprise metric.
        """
        if not self.long_term_memory or self.ltm_optimizer is None:
            return

        self.base_model.train()
        self.long_term_memory.train()
        self.ltm_optimizer.zero_grad()

        logits, values, aux_loss = self.base_model.forward(
            x_batch, ltm_override=self.long_term_memory
        )

        loss_policy = self.policy_loss_fn(
            logits.view(-1, logits.size(-1)), y_batch.view(-1)
        )

        # Calculate value loss (encouraging the model to predict high values)
        # We train the value head to predict 1.0 for any given sequence.
        target_values = torch.ones_like(values)
        loss_value = self.value_loss_fn(values, target_values)

        # Total loss is what drives the LTM update
        total_loss = loss_policy + loss_value
        if aux_loss is not None:
            total_loss += aux_loss

        # Backward pass to compute gradients for LTM
        total_loss.backward()

        # Update LTM based on surprise
        self._update_ltm_and_calc_surprise()

        # Update metrics
        self.metrics.value_score_sum += torch.mean(values).item()
        self.metrics.experience_count += 1

        self.ltm_optimizer.zero_grad()

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
        self, prompt_tokens: torch.Tensor, max_new_tokens=50
    ) -> torch.Tensor:
        """
        Generates a response based on a prompt.
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
        )
        return self.base_model.generate(generate_input)
