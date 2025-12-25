"""
This module contains the Trainer class, which encapsulates the core training logic.
"""
import logging
import os
import time

import numpy as np

from agent_manager import AgentManager
from config import Config
from nn_components.lr_scheduler import cosine_decay_with_warmup
from optimizer import clip_gradients
from train import TrainingData, TrainingComponents
from utils import get_batches, save_checkpoint


class Trainer:
    """
    Encapsulates the training and validation logic.
    """

    def __init__(self, config: Config, components: TrainingComponents, data: TrainingData):
        self.config = config
        self.model = components.model
        self.optimizer = components.optimizer
        self.loss_fn = components.loss_fn
        self.data = data
        self.max_norm = config.optimizer.max_norm

    def run_validation(self):
        """Runs validation on the model."""
        self.model.eval()
        total_loss, num_batches = 0, 0
        batch_iterator = get_batches(self.data.val_data, self.config.evolution.batch_size,
                                     self.config.evolution.seq_len)
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

        batch_iterator = get_batches(self.data.train_data, evo_config.batch_size,
                                     evo_config.seq_len)
        num_batches = len(self.data.train_data) // (evo_config.batch_size *
                                                  evo_config.seq_len)
        training_steps = ((num_batches // evo_config.gradient_accumulation_steps) *
                          evo_config.pretrain_epochs)

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
                new_lr = cosine_decay_with_warmup(
                    current_step, training_steps, max_lr, **scheduler_config.model_dump())
                self.optimizer.lr = new_lr

                params_with_grads = {
                    f"{name}.{k}": (v[0], v[1])
                    for name, layer in self.model.get_named_params().items()
                    if hasattr(layer, 'get_trainable_params')
                    for k, v in layer.get_trainable_params().items()
                }
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

        agent_manager = AgentManager(base_model=self.model,
                                     num_agents=evo_config.num_agents)

        logging.info("Specializing %d agents...", evo_config.num_agents)
        agent_manager.specialize_agents_on_dataset(
            full_data=self.data.train_data,
            evo_config=evo_config,
            steps_per_agent=10
        )

        logging.info("Evaluating and selecting best agents...")
        best_agents = agent_manager.collaborative_evaluation(
            evaluation_data=self.data.val_data[:50],
            tokenizer=self.data.tokenizer,
            top_k=evo_config.num_survivors
        )
        if not best_agents:
            logging.warning("No suitable agents found for merging. Skipping merge.")
            return time.time() - start_time

        logging.info("Merging LTM from %d best agents into base model...",
                     len(best_agents))
        agent_manager.merge_agents(best_agents)

        epoch_time = time.time() - start_time
        logging.info("--- Evolution cycle finished in %.2fs ---", epoch_time)
        return epoch_time

    def run_training(self, model_dir, state):
        """Executes the main training loop, including pre-training and evolution phases."""
        pretrain_epochs = self.config.evolution.pretrain_epochs
        total_epochs = pretrain_epochs + self.config.evolution.evolution_epochs
        logging.info("Starting training loop for %d total epochs.", total_epochs)

        for epoch in range(state.epoch, total_epochs):
            is_pretrain = epoch < pretrain_epochs
            phase = "Pre-training" if is_pretrain else "Evolution"
            phase_epoch = epoch if is_pretrain else epoch - pretrain_epochs
            phase_total_epochs = (
                pretrain_epochs if is_pretrain else self.config.evolution.evolution_epochs)

            logging.info("\n--- %s Epoch %d/%d ---", phase, phase_epoch + 1, phase_total_epochs)

            avg_loss, epoch_time = None, 0.0
            if is_pretrain:
                avg_loss, epoch_time, state.current_step = self.train_pretrain_epoch(
                    state.current_step)
            else:
                epoch_time = self.run_evolution_cycle()

            val_loss = self.run_validation()
            log_parts = [f"    - Validation Loss: {val_loss:.4f}",
                         f"    - Epoch Time: {epoch_time:.2f}s"]
            if avg_loss is not None:
                log_parts.insert(0, f"    - Average Loss: {avg_loss:.4f}")
            if is_pretrain:
                log_parts.append(f"    - Learning Rate: {self.optimizer.lr:.6f}")
            logging.info("\n".join(log_parts))

            if val_loss < state.best_val_loss:
                state.best_val_loss, state.epochs_no_improve = val_loss, 0
                self.model.save_weights(os.path.join(model_dir, 'best_model.npz'),
                                        self.config.model_dump())
                logging.info("    - New best model saved (Val Loss: %.4f)", state.best_val_loss)
            else:
                state.epochs_no_improve += 1
                logging.info("    - No improvement for %d epochs.", state.epochs_no_improve)

            state.epoch = epoch + 1
            save_checkpoint(self.model, self.optimizer, state.__dict__,
                            self.config.model_dump(),
                            os.path.join(model_dir, 'checkpoint.npz'))

            if state.epochs_no_improve >= self.config.evolution.early_stopping_patience:
                logging.warning("Early stopping triggered. Ending training.")
                break
