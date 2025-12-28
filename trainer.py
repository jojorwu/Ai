"""
PyTorch implementation of the Trainer class, which encapsulates the core training logic.
"""
import copy
import logging
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from agent_manager import AgentManager, SpecializationConfig
from config import Config
from data_loader import \
    get_batches_torch as get_batches  # Assuming a torch version exists


@dataclass
class TrainerConfig:
    """Configuration for the Trainer, adapted for PyTorch."""
    model: nn.Module
    optimizer: Adam
    policy_loss_fn: nn.Module
    value_loss_fn: nn.Module
    tokenizer: 'Tokenizer'
    train_data: list
    val_data: list
    config: Config
    accelerator: 'Accelerator'

class Trainer:
    """
    Encapsulates the training and validation logic using PyTorch and Accelerator.
    """

    def __init__(self, **kwargs):
        self._config = TrainerConfig(**kwargs)

    def run_validation(self) -> float:
        """Runs validation on the model."""
        self._config.model.eval()
        total_loss = 0
        num_batches = 0
        evo_cfg = self._config.config.evolution

        batch_iterator = get_batches(
            self._config.val_data, evo_cfg.batch_size, evo_cfg.seq_len, self._config.accelerator.device
        )

        with torch.no_grad():
            for x, y, _ in batch_iterator:
                logits, _, _ = self._config.model(x)
                loss = self._config.policy_loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                total_loss += loss.item()
                num_batches += 1

        self._config.model.train()
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def train_pretrain_epoch(self) -> tuple[float, float]:
        """Runs one epoch of pre-training."""
        start_time = time.time()
        total_policy_loss = 0
        num_batches = 0

        evo_cfg = self._config.config.evolution
        scheduler_cfg = self._config.config.scheduler

        batch_iterator = get_batches(
            self._config.train_data, evo_cfg.batch_size, evo_cfg.seq_len, self._config.accelerator.device
        )

        # Simple scheduler for now
        scheduler = CosineAnnealingLR(self._config.optimizer, T_max=100) # Placeholder T_max

        self._config.model.train()
        self._config.optimizer.zero_grad()

        for i, (x, y, _) in enumerate(batch_iterator):
            # Forward pass
            logits, _, aux_loss = self._config.model(x)

            # Calculate loss
            policy_loss = self._config.policy_loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss = policy_loss
            if aux_loss is not None:
                total_loss += evo_cfg.moe_aux_loss_coeff * aux_loss

            # Scale loss for gradient accumulation
            loss_scaled = total_loss / evo_cfg.gradient_accumulation_steps

            # Backward pass
            self._config.accelerator.backward(loss_scaled)

            total_policy_loss += policy_loss.item()
            num_batches += 1

            # Gradient accumulation step
            if (i + 1) % evo_cfg.gradient_accumulation_steps == 0:
                # Clip gradients
                if self._config.config.optimizer.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self._config.model.parameters(), self._config.config.optimizer.max_norm
                    )
                # Optimizer step
                self._config.optimizer.step()
                # Zero gradients
                self._config.optimizer.zero_grad()

        scheduler.step()
        avg_loss = total_policy_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        return avg_loss, epoch_time

    def run_evolution_cycle(self):
        """Runs one full cycle of evolution."""
        logging.info("--- Starting new evolution cycle ---")
        start_time = time.time()
        evo_config = self._config.config.evolution
        device = self._config.accelerator.device

        # Create a CPU copy of the model for agent cloning to avoid VRAM issues
        cpu_model = copy.deepcopy(self._config.model).to('cpu')

        agent_manager = AgentManager(
            base_model=cpu_model, num_agents=evo_config.num_agents
        )
        logging.info(f"Specializing {evo_config.num_agents} agents...")
        spec_config = SpecializationConfig(
            full_data=self._config.train_data,
            seq_len=evo_config.seq_len,
            batch_size=evo_config.batch_size,
            steps_per_agent=10
        )
        # Pass the device for agent training batches
        agent_manager.specialize_agents_on_dataset(spec_config, device)

        logging.info("Evaluating and selecting best agents...")
        best_agents = agent_manager.collaborative_evaluation(
            evaluation_data=self._config.val_data[:50], # using a slice for speed
            tokenizer=self._config.tokenizer,
            top_k=evo_config.num_survivors,
            device=device
        )

        if best_agents:
            logging.info(f"Merging LTM from {len(best_agents)} best agents into base model...")
            agent_manager.merge_agents(best_agents)
            # Make sure the base model's LTM is on the correct device after merging
            if self._config.model.long_term_memory:
                self._config.model.long_term_memory.to(device)
        else:
            logging.warning("No suitable agents found for merging. Skipping merge.")

        epoch_time = time.time() - start_time
        logging.info(f"--- Evolution cycle finished in {epoch_time:.2f}s ---")
        return epoch_time
