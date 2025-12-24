"""
This module contains the Trainer class, which encapsulates the core training logic.
"""
import logging
import time

import numpy as np

from agent_manager import AgentManager
from config import Config
from model import Transformer
from nn_components.loss import SoftmaxCrossEntropy
from nn_components.lr_scheduler import cosine_decay_with_warmup
from optimizer import Adam, clip_gradients
from tokenizer import Tokenizer
from utils import get_batches

class Trainer:
    """
    Encapsulates the training and validation logic.
    """
    # pylint: disable=too-many-arguments
    def __init__(self, config: Config, model: Transformer, optimizer: Adam,
                 loss_fn: SoftmaxCrossEntropy, tokenizer: Tokenizer,
                 train_data: list, val_data: list):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data
        self.max_norm = config.optimizer.max_norm

    def run_validation(self):
        """Runs validation on the model."""
        self.model.eval()
        total_loss, num_batches = 0, 0
        batch_iterator = get_batches(self.val_data, self.config.evolution.batch_size, self.config.evolution.seq_len)
        for x, y in batch_iterator:
            mask = np.triu(np.ones((x.shape[1], x.shape[1])), k=1).astype(bool)
            logits, _, _ = self.model.forward(x, mask)
            total_loss += self.loss_fn.forward(logits, y)
            num_batches += 1
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else float('inf')

    # pylint: disable=too-many-locals
    def train_pretrain_epoch(self, current_step):
        """Runs one epoch of pre-training."""
        evo_config, scheduler_config = self.config.evolution, self.config.scheduler
        start_time = time.time()
        total_policy_loss = 0

        batch_iterator = get_batches(self.train_data, evo_config.batch_size, evo_config.seq_len)
        num_batches = len(self.train_data) // (evo_config.batch_size * evo_config.seq_len)
        training_steps = (num_batches // evo_config.gradient_accumulation_steps) * evo_config.pretrain_epochs

        self.model.zero_grad()
        for i, (x, y) in enumerate(batch_iterator):
            self.model.train()
            mask = np.triu(np.ones((x.shape[1], x.shape[1])), k=1).astype(bool)

            logits, _, aux_loss = self.model.forward(x, mask)
            policy_loss = self.loss_fn.forward(logits, y)
            total_loss = policy_loss + evo_config.moe_aux_loss_coeff * aux_loss
            total_policy_loss += total_loss.item()

            dlogits = self.loss_fn.backward()
            self.model.backward(dlogits, np.zeros((x.shape[0], 1)))

            if (i + 1) % evo_config.gradient_accumulation_steps == 0:
                clip_gradients(self.model.get_named_params(flat=False), self.max_norm)

                max_lr = self.optimizer.initial_lr
                new_lr = cosine_decay_with_warmup(current_step, training_steps, max_lr, **scheduler_config.model_dump())
                self.optimizer.lr = new_lr

                params_with_grads = {f"{name}.{k}": v for name, layer in self.model.get_named_params().items()
                                     if hasattr(layer, 'get_trainable_params')
                                     for k, v in layer.get_trainable_params().items()}
                self.optimizer.step(params_with_grads)

                self.model.zero_grad()
                current_step += 1

        avg_loss = total_policy_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        return avg_loss, epoch_time, current_step

    def run_evolution_cycle(self):
        """Runs one full cycle of evolution."""
        logging.info("--- Starting new evolution cycle ---")
        start_time = time.time()
        evo_config = self.config.evolution

        agent_manager = AgentManager(base_model=self.model, num_agents=evo_config.num_agents)

        logging.info(f"Specializing {evo_config.num_agents} agents...")
        agent_manager.specialize_agents_on_dataset(
            full_data=self.train_data,
            tokenizer=self.tokenizer,
            seq_len=evo_config.seq_len,
            batch_size=evo_config.batch_size,
            steps_per_agent=10
        )

        logging.info("Evaluating and selecting best agents...")
        best_agents = agent_manager.collaborative_evaluation(
            evaluation_data=self.val_data[:50],
            tokenizer=self.tokenizer,
            top_k=evo_config.num_survivors
        )
        if not best_agents:
            logging.warning("No suitable agents found for merging. Skipping merge.")
            return time.time() - start_time

        logging.info(f"Merging LTM from {len(best_agents)} best agents into base model...")
        agent_manager.merge_agents(best_agents)

        epoch_time = time.time() - start_time
        logging.info(f"--- Evolution cycle finished in {epoch_time:.2f}s ---")
        return epoch_time
