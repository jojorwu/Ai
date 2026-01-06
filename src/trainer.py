"""
PyTorch implementation of the Trainer class, which encapsulates the core training logic.
"""
import logging
import time
from dataclasses import dataclass

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.agent.agent_manager import AgentManager, SpecializationConfig
from src.config import TrainConfig, TransformerConfig
from src.data.data_loader import \
    get_batches_torch as get_batches  # Assuming a torch version exists
from src.model.loss import cross_entropy_with_label_smoothing
from src.model.model import Transformer


def create_trainer(
    config: TrainConfig,
    data_components: "DataComponents",
    accelerator: "Accelerator",
    load_in_4bit: bool = False,
) -> "Trainer":
    """Factory function to create a Trainer instance."""
    return Trainer(config, data_components, accelerator, load_in_4bit)


@dataclass
class DataComponents:
    """Data-related components for training."""
    tokenizer: 'Tokenizer'
    train_data: list
    val_data: list


class Trainer:
    """
    Encapsulates the training and validation logic using PyTorch and Accelerator.
    """

    def __init__(
        self,
        config: TrainConfig,
        data_components: "DataComponents",
        accelerator: "Accelerator",
        load_in_4bit: bool = False,
    ):
        self.config = config
        self.data_components = data_components
        self.accelerator = accelerator

        # Setup model, optimizer, and scheduler
        self.model = self._setup_model(load_in_4bit)
        (
            self.optimizer,
            self.scheduler,
            self.value_loss_fn,
        ) = self._setup_optimizer_and_scheduler()

    def _setup_model(self, load_in_4bit: bool) -> nn.Module:
        """Initializes the Transformer model, applying LoRA if configured."""
        transformer_config = TransformerConfig(
            vocab_size=self.data_components.tokenizer.vocab_size,
            model=self.config.model,
            vision=self.config.vision,
            ltm=self.config.ltm,
            tokenizer=self.data_components.tokenizer,
        )
        model = Transformer(transformer_config, load_in_4bit=load_in_4bit)

        if load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        if self.config.lora:
            logging.info("Applying LoRA configuration...")
            lora_config = LoraConfig(
                r=self.config.lora.r,
                lora_alpha=self.config.lora.lora_alpha,
                target_modules=self.config.lora.target_modules,
                lora_dropout=self.config.lora.lora_dropout,
                bias=self.config.lora.bias,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            logging.info("LoRA applied successfully.")
            model.print_trainable_parameters()
        return model

    def _setup_optimizer_and_scheduler(self):
        """Initializes the optimizer and learning rate scheduler."""
        optimizer = Adam(self.model.parameters(), lr=self.config.optimizer.learning_rate)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.config.scheduler.training_steps
        )
        value_loss_fn = nn.MSELoss()

        # Prepare all components with the accelerator.
        model, optimizer, scheduler, value_loss_fn = self.accelerator.prepare(
            self.model, optimizer, scheduler, value_loss_fn
        )
        # After `prepare`, the model might be a wrapper, so we need to update our reference.
        self.model = model
        return optimizer, scheduler, value_loss_fn

    def get_model(self) -> nn.Module:
        """Returns the underlying model."""
        return self.model

    def get_learning_rate(self) -> float:
        """Returns the current learning rate from the scheduler."""
        return self.scheduler.get_last_lr()[0]

    def run_validation(self) -> float:
        """Runs validation on the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        evo_cfg = self.config.evolution
        batch_iterator = get_batches(
            self.data_components.val_data,
            evo_cfg.batch_size,
            evo_cfg.seq_len,
            self.accelerator.device,
        )
        with torch.no_grad():
            for x, y, _ in batch_iterator:
                logits, _, _ = self.model(x)
                loss = cross_entropy_with_label_smoothing(
                    logits,
                    y,
                    smoothing=evo_cfg.label_smoothing,
                    vocab_size=self.data_components.tokenizer.vocab_size,
                )
                total_loss += loss.item()
                num_batches += 1
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def _run_training_step(self, x, y):
        """Runs a single training step, computes loss, and backpropagates."""
        logits, _, aux_loss = self.model(x)
        policy_loss = cross_entropy_with_label_smoothing(
            logits,
            y,
            smoothing=self.config.evolution.label_smoothing,
            vocab_size=self.data_components.tokenizer.vocab_size,
        )
        total_loss = policy_loss + (
            self.config.evolution.moe_aux_loss_coeff * aux_loss
            if aux_loss
            else 0
        )
        loss_scaled = (
            total_loss / self.config.evolution.gradient_accumulation_steps
        )
        self.accelerator.backward(loss_scaled)
        return policy_loss.item()

    def _perform_update_step(self):
        """Performs a model update, including gradient clipping and optimizer step."""
        if self.config.optimizer.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.optimizer.max_norm
            )
        self.optimizer.step()
        self.optimizer.zero_grad()

    def train_pretrain_epoch(self) -> tuple[float, float]:
        """Runs one epoch of pre-training."""
        start_time = time.time()
        total_policy_loss = 0
        num_batches = 0
        evo_cfg = self.config.evolution
        batch_iterator = get_batches(
            self.data_components.train_data,
            evo_cfg.batch_size,
            evo_cfg.seq_len,
            self.accelerator.device,
        )
        self.model.train()
        self.optimizer.zero_grad()

        for i, (x, y, _) in enumerate(batch_iterator):
            total_policy_loss += self._run_training_step(x, y)
            num_batches += 1
            if (i + 1) % evo_cfg.gradient_accumulation_steps == 0:
                self._perform_update_step()

        self.scheduler.step()
        avg_loss = total_policy_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        return avg_loss, epoch_time

    def _specialize_agents(self, agent_manager: AgentManager):
        """Manages the agent specialization phase."""
        logging.info("Specializing %d agents...", self.config.evolution.num_agents)
        spec_config = SpecializationConfig(
            full_data=self.data_components.train_data,
            seq_len=self.config.evolution.seq_len,
            batch_size=self.config.evolution.batch_size,
            steps_per_agent=self.config.evolution.agent_specialization_steps,
        )
        agent_manager.specialize_agents_on_dataset(
            spec_config, self.accelerator.device
        )

    def _evaluate_and_merge_agents(self, agent_manager: AgentManager):
        """Handles the collaborative evaluation and merging of the best agents."""
        logging.info("Evaluating and selecting best agents...")
        evaluation_data = self.data_components.val_data[
            : self.config.evolution.evaluation_data_size
        ]
        best_agents = agent_manager.collaborative_evaluation(
            evaluation_data=evaluation_data,
            tokenizer=self.data_components.tokenizer,
            top_k=self.config.evolution.num_survivors,
            device=self.accelerator.device,
        )

        if best_agents:
            logging.info(
                "Merging LTM from %d best agents into base model...",
                len(best_agents),
            )
            agent_manager.merge_agents(best_agents)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            if unwrapped_model.layers.long_term_memory:
                unwrapped_model.layers.long_term_memory.to(self.accelerator.device)
        else:
            logging.warning("No suitable agents found for merging. Skipping merge.")

    def run_evolution_cycle(self) -> float:
        """Runs one full cycle of evolution: specialization, evaluation, and merging."""
        logging.info("--- Starting new evolution cycle ---")
        start_time = time.time()

        agent_manager = AgentManager(
            base_model=self.model,
            num_agents=self.config.evolution.num_agents,
            ltm_config=self.config.ltm,
            accelerator=self.accelerator,
        )

        self._specialize_agents(agent_manager)
        self._evaluate_and_merge_agents(agent_manager)

        epoch_time = time.time() - start_time
        logging.info("--- Evolution cycle finished in %.2fs ---", epoch_time)
        return epoch_time
