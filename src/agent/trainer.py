"""
Trainer for the Agent class.
"""
import logging
import math
import torch

from src.agent.agent import Agent
from src.agent.update_policy import SurpriseUpdatePolicy, UpdatePolicy
from src.utils.training import calculate_gradient_norm


class AgentTrainer:
    """
    Handles the training process for a single agent, including the "experience" loop.
    """

    def __init__(
        self, agent: Agent, update_policy: UpdatePolicy | None = None
    ) -> None:
        self.agent = agent
        self.update_policy = update_policy or SurpriseUpdatePolicy()

    def _update_ltm_and_calc_surprise(self) -> float:
        """
        Calculates the gradient norm for LTM parameters ("surprise") and,
        if it exceeds a threshold (determined by policy), performs an optimizer step.
        """
        if not self.agent.long_term_memory:
            return 0.0

        surprise = calculate_gradient_norm(self.agent.long_term_memory.parameters())

        # Only update metrics and perform optimizer step if surprise is finite.
        if math.isfinite(surprise):
            self.agent.metrics.total_surprise += surprise

            if self.update_policy.should_update(surprise, self.agent.ltm_config):
                self.agent.ltm_optimizer.step()
        else:
            logging.warning(
                "Agent %s: Non-finite surprise (%.4f) detected. Skipping optimizer step.",
                self.agent.agent_id, surprise
            )

        return surprise

    def experience(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> None:
        """
        The process of an agent gaining "experience" in a batch training mode.
        This method updates the agent's LTM based on the surprise metric.

        Args:
            x_batch: Input token indices of shape [batch, seq_len].
            y_batch: Target token indices of shape [batch, seq_len].
        """
        if not self.agent.long_term_memory or self.agent.ltm_optimizer is None:
            return

        self.agent.base_model.train()
        self.agent.long_term_memory.train()
        self.agent.ltm_optimizer.zero_grad()

        logits, values, aux_loss, _ = self.agent.base_model.forward(
            x_batch, ltm_override=self.agent.long_term_memory
        )

        loss_policy = self.agent.policy_loss_fn(
            logits.view(-1, logits.size(-1)), y_batch.view(-1)
        )

        # Calculate value loss (encouraging the model to predict high values)
        target_values = torch.ones_like(values)
        loss_value = self.agent.value_loss_fn(values, target_values)

        # Total loss is what drives the LTM update
        total_loss = loss_policy + loss_value
        if aux_loss is not None:
            total_loss += aux_loss

        # Check for NaN/Inf in loss to prevent weight corruption and autograd errors
        if not torch.isfinite(total_loss):
            logging.warning("Agent %s encountered non-finite loss (%.4f). Skipping update.", self.agent.agent_id, total_loss.item())
            return

        # Compute gradients ONLY for LTM parameters to avoid touching the shared base_model's gradients.
        # This is essential for thread-safety during parallel agent training.
        ltm_params = list(self.agent.long_term_memory.parameters())
        grads = torch.autograd.grad(total_loss, ltm_params, allow_unused=True)

        for param, grad in zip(ltm_params, grads):
            if grad is not None:
                param.grad = grad

        # Perform gradient clipping for LTM parameters
        torch.nn.utils.clip_grad_norm_(ltm_params, max_norm=1.0)

        # Update LTM based on surprise (handles its own finite checks)
        self._update_ltm_and_calc_surprise()

        # Update metrics only if the value is finite
        mean_value = torch.mean(values).item()
        if math.isfinite(mean_value):
            self.agent.metrics.value_score_sum += mean_value
            self.agent.metrics.experience_count += 1
        else:
            logging.warning(
                "Agent %s: Non-finite value detected during experience. Skipping metric update.",
                self.agent.agent_id
            )
