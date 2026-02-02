"""
Implements the runners for different training phases.
"""
from __future__ import annotations
import logging
import time
from typing import TYPE_CHECKING, Tuple, List, Any

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.agent.dataclasses import SpecializationConfig
from src.data.data_loader import get_batches_torch as get_batches
from src.training.loss import cross_entropy_with_label_smoothing

if TYPE_CHECKING:
    from accelerate import Accelerator
    from src.agent.agent_manager import AgentManager
    from src.config.training_config import EvolutionConfig, OptimizerConfig
    from src.config.hardware_config import HardwareConfig
    from src.model.model import Transformer


def calculate_loss(
    model: Transformer, x: torch.Tensor, y: torch.Tensor, evolution_config: EvolutionConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Common loss calculation logic for Transformer model with MoE support.

    Args:
        model: The Transformer model.
        x: Input token IDs.
        y: Target token IDs.
        evolution_config: Configuration for evolution (including MoE settings).

    Returns:
        A tuple of (total_loss, policy_loss).
    """
    outputs = model(x)
    policy_loss = cross_entropy_with_label_smoothing(
        outputs.logits,
        y,
        smoothing=evolution_config.label_smoothing,
    )
    total_loss = policy_loss + (
        evolution_config.moe_aux_loss_coeff * outputs.aux_loss
        if outputs.aux_loss is not None
        else 0
    )
    return total_loss, policy_loss


class EvolutionRunner:
    """
    Handles the evolutionary training cycle.

    This includes agent specialization, collaborative evaluation, and merging
    the best agents' knowledge back into the base model.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        model: Transformer,
        train_data: Any,
        val_data: Any,
        tokenizer: Any,
        evolution_config: EvolutionConfig,
        agent_manager: AgentManager,
    ) -> None:
        """
        Initializes the EvolutionRunner.

        Args:
            accelerator: Accelerator object.
            model: Base Transformer model.
            train_data: Training dataset.
            val_data: Validation dataset.
            tokenizer: Tokenizer for decoding.
            evolution_config: Configuration for evolution.
            agent_manager: Manager for the agent population.
        """
        self.accelerator = accelerator
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.tokenizer = tokenizer
        self.evolution_config = evolution_config
        self.agent_manager = agent_manager

    def run(self) -> float:
        """
        Runs one full cycle of evolution.

        Returns:
            The time taken for the evolution cycle in seconds.
        """
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
        with torch.inference_mode():
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
                self.model.layers.long_term_memory.to(device, non_blocking=True)
        else:
            logging.warning("No suitable agents found for merging. Skipping merge.")
        epoch_time = time.time() - start_time
        logging.info("--- Evolution cycle finished in %.2fs ---", epoch_time)
        return epoch_time


class PretrainingRunner:
    """
    Handles the pre-training epoch.

    Performs standard supervised training on the provided dataset.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        model: Transformer,
        optimizer: Adam,
        scheduler: Any,
        train_data: Any,
        tokenizer: Any,
        evolution_config: EvolutionConfig,
        optimizer_config: OptimizerConfig,
        hardware_config: HardwareConfig | None = None,
    ) -> None:
        """
        Initializes the PretrainingRunner.

        Args:
            accelerator: Accelerator object.
            model: Transformer model.
            optimizer: Optimizer for the model.
            scheduler: Learning rate scheduler.
            train_data: Training dataset.
            tokenizer: Tokenizer.
            evolution_config: Configuration for evolution.
            optimizer_config: Configuration for the optimizer.
            hardware_config: Hardware settings.
        """
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_data
        self.tokenizer = tokenizer
        self.evolution_config = evolution_config
        self.optimizer_config = optimizer_config
        self.hardware_config = hardware_config

    def _run_training_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Runs a single training step.

        Args:
            x: Input tokens.
            y: Target tokens.

        Returns:
            The policy loss value as a float.
        """
        total_loss, policy_loss = calculate_loss(
            self.model, x, y, self.evolution_config
        )
        loss_scaled = total_loss / self.evolution_config.gradient_accumulation_steps
        self.accelerator.backward(loss_scaled)
        return policy_loss.item()

    def run(self) -> Tuple[float, float]:
        """
        Runs one epoch of pre-training.

        Returns:
            A tuple of (average_policy_loss, epoch_time).
        """
        start_time = time.time()
        total_policy_loss = 0.0
        num_batches = 0
        batch_iterator = get_batches(
            self.train_data,
            self.evolution_config.batch_size,
            self.evolution_config.seq_len,
            self.accelerator.device,
            pin_memory=self.hardware_config.pin_memory if self.hardware_config else False,
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
        avg_loss = total_policy_loss / num_batches if num_batches > 0 else 0.0
        epoch_time = time.time() - start_time
        return avg_loss, epoch_time


class ValidationRunner:
    """
    Handles the validation loop.

    Evaluates the model on a validation dataset without updating weights.
    """

    def __init__(
        self,
        model: Transformer,
        val_data: Any,
        tokenizer: Any,
        evolution_config: EvolutionConfig,
        accelerator: Accelerator,
        hardware_config: HardwareConfig | None = None,
    ) -> None:
        """
        Initializes the ValidationRunner.

        Args:
            model: Transformer model.
            val_data: Validation dataset.
            tokenizer: Tokenizer.
            evolution_config: Configuration for evolution.
            accelerator: Accelerator object.
            hardware_config: Hardware settings.
        """
        self.model = model
        self.val_data = val_data
        self.tokenizer = tokenizer
        self.evolution_config = evolution_config
        self.accelerator = accelerator
        self.hardware_config = hardware_config

    def run(self) -> float:
        """
        Runs validation on the model.

        Returns:
            The average policy loss on the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        batch_iterator = get_batches(
            self.val_data,
            self.evolution_config.batch_size,
            self.evolution_config.seq_len,
            self.accelerator.device,
            pin_memory=self.hardware_config.pin_memory if self.hardware_config else False,
        )
        with torch.inference_mode():
            for x, y, _ in batch_iterator:
                _, policy_loss = calculate_loss(
                    self.model, x, y, self.evolution_config
                )
                total_loss += policy_loss.item()
                num_batches += 1
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else float("inf")
