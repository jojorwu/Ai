"""
Trainer for the Agent class.
"""
import torch

from src.agent.agent import Agent


class AgentTrainer:
    """
    Handles the training process for a single agent, including the "experience" loop.
    """

    def __init__(self, agent: Agent):
        self.agent = agent

    def _update_ltm_and_calc_surprise(self) -> float:
        """
        Calculates the gradient norm for LTM parameters ("surprise") and,
        if it exceeds a threshold, performs an optimizer step.
        """
        if not self.agent.long_term_memory:
            return 0.0

        grad_tensors = [
            p.grad.detach().flatten()
            for p in self.agent.long_term_memory.parameters()
            if p.grad is not None
        ]

        if not grad_tensors:
            return 0.0

        surprise = torch.norm(torch.cat(grad_tensors)).item()
        self.agent.metrics.total_surprise += surprise

        if surprise > self.agent.ltm_config.surprise_threshold:
            self.agent.ltm_optimizer.step()

        return surprise

    def experience(self, x_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        The process of an agent gaining "experience" in a batch training mode.
        This method updates the agent's LTM based on the surprise metric.
        """
        if not self.agent.long_term_memory or self.agent.ltm_optimizer is None:
            return

        self.agent.base_model.train()
        self.agent.long_term_memory.train()
        self.agent.ltm_optimizer.zero_grad()

        logits, values, aux_loss = self.agent.base_model.forward(
            x_batch, ltm_override=self.agent.long_term_memory
        )

        loss_policy = self.agent.policy_loss_fn(
            logits.view(-1, logits.size(-1)), y_batch.view(-1)
        )

        # Calculate value loss (encouraging the model to predict high values)
        # We train the value head to predict 1.0 for any given sequence.
        target_values = torch.ones_like(values)
        loss_value = self.agent.value_loss_fn(values, target_values)

        # Total loss is what drives the LTM update
        total_loss = loss_policy + loss_value
        if aux_loss is not None:
            total_loss += aux_loss

        # Backward pass to compute gradients for LTM
        total_loss.backward()

        # Update LTM based on surprise
        self._update_ltm_and_calc_surprise()

        # Update metrics
        self.agent.metrics.value_score_sum += torch.mean(values).item()
        self.agent.metrics.experience_count += 1

        self.agent.ltm_optimizer.zero_grad()
