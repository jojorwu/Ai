"""
Implements the runners for different training phases.
"""
import logging
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.agent.agent_manager import AgentManager
from src.agent.dataclasses import SpecializationConfig
from src.agent.evaluator import CollaborativeEvaluator
from src.config.core import TrainConfig
from src.data.data_loader import get_batches_torch as get_batches
from src.training.loss import cross_entropy_with_label_smoothing


def calculate_loss(model, x, y, evolution_config):
    """
    Common loss calculation logic for Transformer model with MoE support.
    """
    logits, _, aux_loss = model(x)
    policy_loss = cross_entropy_with_label_smoothing(
        logits,
        y,
        smoothing=evolution_config.label_smoothing,
    )
    total_loss = policy_loss + (
        evolution_config.moe_aux_loss_coeff * aux_loss if aux_loss else 0
    )
    return total_loss, policy_loss


class EvolutionRunner:  # pylint: disable=too-few-public-methods
    """Handles the evolutionary cycle."""

    def __init__(
        self,
        accelerator,
        model,
        train_data,
        val_data,
        tokenizer,
        evolution_config,
        agent_manager: AgentManager,
    ):
        self.accelerator = accelerator
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.tokenizer = tokenizer
        self.evolution_config = evolution_config
        self.agent_manager = agent_manager

    def run(self):
        """Runs one full cycle of evolution."""
        logging.info("--- Starting new evolution cycle ---")
        start_time = time.time()
        device = self.accelerator.device

        logging.info("Specializing %d agents...", self.evolution_config.num_agents)
        spec_config = SpecializationConfig(
            full_data=self.train_data,
            seq_len=self.evolution_config.seq_len,
            batch_size=self.evolution_config.batch_size,
            steps_per_agent=10,
        )
        self.agent_manager.specialize_agents_on_dataset(spec_config, device)

        logging.info("Evaluating and selecting best agents...")
        best_agents = self.agent_manager.collaborative_evaluation(
            evaluation_data=self.val_data[:50],
            tokenizer=self.tokenizer,
            top_k=self.evolution_config.num_survivors,
            device=device,
        )

        if best_agents:
            logging.info(
                "Merging LTM from %d best agents into base model...",
                len(best_agents),
            )
            self.agent_manager.merge_agents(best_agents)
            if self.model.layers.long_term_memory:
                self.model.layers.long_term_memory.to(device)
        else:
            logging.warning("No suitable agents found for merging. Skipping merge.")
        epoch_time = time.time() - start_time
        logging.info("--- Evolution cycle finished in %.2fs ---", epoch_time)
        return epoch_time


class PretrainingRunner:  # pylint: disable=too-few-public-methods
    """Handles the pre-training epoch."""

    def __init__(
        self,
        accelerator,
        model,
        optimizer,
        scheduler,
        train_data,
        tokenizer,
        evolution_config,
        optimizer_config,
    ):
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_data
        self.tokenizer = tokenizer
        self.evolution_config = evolution_config
        self.optimizer_config = optimizer_config

    def _run_training_step(self, x, y):
        """Runs a single training step."""
        total_loss, policy_loss = calculate_loss(
            self.model, x, y, self.evolution_config
        )
        loss_scaled = total_loss / self.evolution_config.gradient_accumulation_steps
        self.accelerator.backward(loss_scaled)
        return policy_loss.item()

    def run(self) -> tuple[float, float]:
        """Runs one epoch of pre-training."""
        start_time = time.time()
        total_policy_loss = 0
        num_batches = 0
        batch_iterator = get_batches(
            self.train_data,
            self.evolution_config.batch_size,
            self.evolution_config.seq_len,
            self.accelerator.device,
        )
        self.model.train()
        self.optimizer.zero_grad()
        for i, (x, y, _) in enumerate(batch_iterator):
            total_policy_loss += self._run_training_step(x, y)
            num_batches += 1
            if (i + 1) % self.evolution_config.gradient_accumulation_steps == 0:
                if self.optimizer_config.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.optimizer_config.max_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.scheduler.step()
        avg_loss = total_policy_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        return avg_loss, epoch_time


class ValidationRunner:  # pylint: disable=too-few-public-methods
    """Handles the validation loop."""

    def __init__(self, model, val_data, tokenizer, evolution_config, accelerator):
        self.model = model
        self.val_data = val_data
        self.tokenizer = tokenizer
        self.evolution_config = evolution_config
        self.accelerator = accelerator

    def run(self) -> float:
        """Runs validation on the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        batch_iterator = get_batches(
            self.val_data,
            self.evolution_config.batch_size,
            self.evolution_config.seq_len,
            self.accelerator.device,
        )
        with torch.inference_mode():
            for x, y, _ in batch_iterator:
                _, policy_loss = calculate_loss(
                    self.model, x, y, self.evolution_config
                )
                total_loss += policy_loss.item()
                num_batches += 1
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else float('inf')
