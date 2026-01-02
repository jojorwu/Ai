"""
PyTorch implementation of the Trainer class, which encapsulates the core training logic.
"""
import logging
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.agent.agent_manager import AgentManager, SpecializationConfig
from src.config import Config, TransformerConfig
from src.data.data_loader import \
    get_batches_torch as get_batches  # Assuming a torch version exists
from src.model.loss import cross_entropy_with_label_smoothing
from src.model.model import Transformer


def create_trainer(
    config: Config,
    data_components: "DataComponents",
    accelerator: "Accelerator",
    load_in_4bit: bool = False,
) -> "Trainer":
    """Initializes and returns a Trainer instance."""
    transformer_config = TransformerConfig(
        vocab_size=data_components.tokenizer.vocab_size,
        model=config.model,
        vision=config.vision,
        ltm=config.ltm,
        tokenizer=data_components.tokenizer,
    )
    model = Transformer(transformer_config, load_in_4bit=load_in_4bit)
    optimizer = Adam(model.parameters(), lr=config.optimizer.learning_rate)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.scheduler.training_steps
    )
    value_loss_fn = nn.MSELoss()

    model, optimizer, scheduler, value_loss_fn = accelerator.prepare(
        model, optimizer, scheduler, value_loss_fn
    )

    training_components = TrainingComponents(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        value_loss_fn=value_loss_fn,
    )
    trainer_config = TrainerConfig(
        components=training_components,
        data=data_components,
        config=config,
        accelerator=accelerator,
    )
    return Trainer(trainer_config)


@dataclass
class TrainingComponents:
    """Core components for training."""
    model: nn.Module
    optimizer: Adam
    scheduler: CosineAnnealingLR
    value_loss_fn: nn.Module

@dataclass
class DataComponents:
    """Data-related components for training."""
    tokenizer: 'Tokenizer'
    train_data: list
    val_data: list

@dataclass
class TrainerConfig:
    """Configuration for the Trainer, adapted for PyTorch."""
    components: TrainingComponents
    data: DataComponents
    config: Config
    accelerator: 'Accelerator'


class Trainer:
    """
    Encapsulates the training and validation logic using PyTorch and Accelerator.
    """

    def __init__(self, trainer_config: TrainerConfig):
        self._config = trainer_config

    def get_model(self) -> nn.Module:
        """Returns the underlying model."""
        return self._config.components.model

    def run_validation(self) -> float:
        """Runs validation on the model."""
        self._config.components.model.eval()
        total_loss = 0
        num_batches = 0
        evo_cfg = self._config.config.evolution
        batch_iterator = get_batches(
            self._config.data.val_data,
            evo_cfg.batch_size,
            evo_cfg.seq_len,
            self._config.accelerator.device,
        )
        with torch.no_grad():
            for x, y, _ in batch_iterator:
                logits, _, _ = self._config.components.model(x)
                loss = cross_entropy_with_label_smoothing(
                    logits,
                    y,
                    smoothing=evo_cfg.label_smoothing,
                    vocab_size=self._config.data.tokenizer.vocab_size,
                )
                total_loss += loss.item()
                num_batches += 1
        self._config.components.model.train()
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def _run_training_step(self, x, y, evo_cfg):
        """Runs a single training step."""
        logits, _, aux_loss = self._config.components.model(x)
        policy_loss = cross_entropy_with_label_smoothing(
            logits,
            y,
            smoothing=evo_cfg.label_smoothing,
            vocab_size=self._config.data.tokenizer.vocab_size,
        )
        total_loss = policy_loss + (
            evo_cfg.moe_aux_loss_coeff * aux_loss if aux_loss else 0
        )
        loss_scaled = total_loss / evo_cfg.gradient_accumulation_steps
        self._config.accelerator.backward(loss_scaled)
        return policy_loss.item()

    def train_pretrain_epoch(self) -> tuple[float, float]:
        """Runs one epoch of pre-training."""
        start_time = time.time()
        total_policy_loss = 0
        num_batches = 0
        evo_cfg = self._config.config.evolution
        batch_iterator = get_batches(
            self._config.data.train_data,
            evo_cfg.batch_size,
            evo_cfg.seq_len,
            self._config.accelerator.device,
        )
        self._config.components.model.train()
        self._config.components.optimizer.zero_grad()
        for i, (x, y, _) in enumerate(batch_iterator):
            total_policy_loss += self._run_training_step(x, y, evo_cfg)
            num_batches += 1
            if (i + 1) % evo_cfg.gradient_accumulation_steps == 0:
                if self._config.config.optimizer.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self._config.components.model.parameters(),
                        self._config.config.optimizer.max_norm,
                    )
                self._config.components.optimizer.step()
                self._config.components.optimizer.zero_grad()
        self._config.components.scheduler.step()
        avg_loss = total_policy_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        return avg_loss, epoch_time

    def run_evolution_cycle(self):
        """Runs one full cycle of evolution."""
        logging.info("--- Starting new evolution cycle ---")
        start_time = time.time()
        evo_config = self._config.config.evolution
        device = self._config.accelerator.device
        agent_manager = AgentManager(
            base_model=self._config.components.model,
            num_agents=evo_config.num_agents,
            accelerator=self._config.accelerator,
        )
        logging.info("Specializing %d agents...", evo_config.num_agents)
        spec_config = SpecializationConfig(
            full_data=self._config.data.train_data,
            seq_len=evo_config.seq_len,
            batch_size=evo_config.batch_size,
            steps_per_agent=10,
        )
        agent_manager.specialize_agents_on_dataset(spec_config, device)
        logging.info("Evaluating and selecting best agents...")
        best_agents = agent_manager.collaborative_evaluation(
            evaluation_data=self._config.data.val_data[:50],
            tokenizer=self._config.data.tokenizer,
            top_k=evo_config.num_survivors,
            device=device,
        )
        if best_agents:
            logging.info(
                "Merging LTM from %d best agents into base model...",
                len(best_agents),
            )
            agent_manager.merge_agents(best_agents)
            if self._config.components.model.layers.long_term_memory:
                self._config.components.model.layers.long_term_memory.to(device)
        else:
            logging.warning("No suitable agents found for merging. Skipping merge.")
        epoch_time = time.time() - start_time
        logging.info("--- Evolution cycle finished in %.2fs ---", epoch_time)
        return epoch_time
