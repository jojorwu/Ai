"""
This module contains the Trainer class, which encapsulates the core training logic.
"""
import logging
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from agent_manager import AgentManager, SpecializationConfig
from config import Config
from data_loader import get_batches
from nn_components.lr_scheduler import cosine_decay_with_warmup
from optimizer import clip_gradients


@dataclass
class TrainerConfig:
    """Configuration for the Trainer."""
    model: 'Transformer'
    optimizer: 'Adam'
    loss_fn: 'SoftmaxCrossEntropy'
    tokenizer: 'Tokenizer'
    train_data: list
    val_data: list
    config: Config


@dataclass
class PretrainStepContext:
    """Context for a single pre-training step."""
    x: np.ndarray
    y: np.ndarray
    current_step: int
    training_steps: int


class Trainer:
    """
    Encapsulates the training and validation logic.
    """

    def __init__(self, **kwargs):
        self._config = TrainerConfig(**kwargs)
        self.max_norm = self._config.config.optimizer.max_norm

    def run_validation(self) -> float:
        """Runs validation on the model."""
        self._config.model.eval()
        total_loss, num_batches = 0, 0
        evo_cfg = self._config.config.evolution
        batch_iterator = get_batches(self._config.val_data,
                                     evo_cfg.batch_size, evo_cfg.seq_len)

        for x, y in batch_iterator:
            mask = np.triu(np.ones((x.shape[1], x.shape[1])), k=1).astype(bool)
            forward_input = self._config.model.ForwardPassInput(x=x, ltm_state=0, mask=mask)
            logits, _, _ = self._config.model.forward(forward_input)
            total_loss += self._config.loss_fn.forward(logits, y)
            num_batches += 1
        self._config.model.train()
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def _get_batch_iterator(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Returns a batch iterator for the training data."""
        evo_cfg = self._config.config.evolution
        return get_batches(self._config.train_data,
                           evo_cfg.batch_size, evo_cfg.seq_len)

    def _run_pretrain_step(self, context: PretrainStepContext) -> float:
        """Runs a single pre-training step."""
        evo_cfg = self._config.config.evolution
        self._config.model.train()
        mask = np.triu(np.ones((context.x.shape[1], context.x.shape[1])), k=1).astype(bool)
        forward_input = self._config.model.ForwardPassInput(x=context.x, ltm_state=0, mask=mask)
        logits, _, aux_loss = self._config.model.forward(forward_input)
        policy_loss = self._config.loss_fn.forward(logits, context.y)
        total_loss = policy_loss + evo_cfg.moe_aux_loss_coeff * aux_loss
        dlogits = self._config.loss_fn.backward()
        self._config.model.backward(dlogits, np.zeros((context.x.shape[0], 1)))

        if (context.current_step + 1) % evo_cfg.gradient_accumulation_steps == 0:
            self._update_gradients(context.current_step, context.training_steps)
        return total_loss.item()

    def _update_gradients(self, current_step: int, training_steps: int):
        """Clips gradients, updates learning rate, and steps the optimizer."""
        evo_cfg = self._config.config.evolution
        scheduler_cfg = self._config.config.scheduler
        named_params = self._config.model.get_named_params(flat=False)
        clip_gradients(named_params, self.max_norm)
        max_lr = self._config.optimizer.initial_lr
        new_lr = cosine_decay_with_warmup(
            current_step // evo_cfg.gradient_accumulation_steps,
            training_steps, max_lr, **scheduler_cfg.model_dump())
        self._config.optimizer.lr = new_lr

        params_with_grads = {
            f"{name}.{k}": (v[0], v[1])
            for name, layer in self._config.model.get_named_params().items()
            if hasattr(layer, 'get_trainable_params')
            for k, v in layer.get_trainable_params().items()
        }
        self._config.optimizer.step(params_with_grads)
        self._config.model.zero_grad()

    def train_pretrain_epoch(self) -> tuple[float, float, int]:
        """Runs one epoch of pre-training."""
        start_time = time.time()
        total_policy_loss = 0
        current_step_in_epoch = 0
        batch_iterator = self._get_batch_iterator()
        evo_cfg = self._config.config.evolution
        num_batches = len(self._config.train_data) // (evo_cfg.batch_size * evo_cfg.seq_len)
        training_steps = (num_batches // evo_cfg.gradient_accumulation_steps * evo_cfg.pretrain_epochs)

        self._config.model.zero_grad()
        for i, (x, y) in enumerate(batch_iterator):
            context = PretrainStepContext(x=x, y=y, current_step=i, training_steps=training_steps)
            total_policy_loss += self._run_pretrain_step(context)
            if (i + 1) % evo_cfg.gradient_accumulation_steps == 0:
                current_step_in_epoch += 1

        avg_loss = total_policy_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        return avg_loss, epoch_time, current_step_in_epoch

    def run_evolution_cycle(self):
        """Runs one full cycle of evolution."""
        logging.info("--- Starting new evolution cycle ---")
        start_time = time.time()
        evo_config = self._config.config.evolution

        agent_manager = AgentManager(
            base_model=self._config.model, num_agents=evo_config.num_agents)
        logging.info("Specializing %d agents...", evo_config.num_agents)
        spec_config = SpecializationConfig(
            full_data=self._config.train_data,
            seq_len=evo_config.seq_len,
            batch_size=evo_config.batch_size,
            steps_per_agent=10
        )
        agent_manager.specialize_agents_on_dataset(spec_config)

        logging.info("Evaluating and selecting best agents...")
        best_agents = agent_manager.collaborative_evaluation(
            evaluation_data=self._config.val_data[:50],
            tokenizer=self._config.tokenizer,
            top_k=evo_config.num_survivors
        )
        if best_agents:
            logging.info("Merging LTM from %d best agents into base model...", len(best_agents))
            agent_manager.merge_agents(best_agents)
        else:
            logging.warning("No suitable agents found for merging. Skipping merge.")

        epoch_time = time.time() - start_time
        logging.info("--- Evolution cycle finished in %.2fs ---", epoch_time)
        return epoch_time
