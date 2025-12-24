"""
Main script for agent-centric training of the Transformer model.
"""
import argparse
import json
import logging
import os
import shutil
import time

from backend import set_backend
from config import Config
from data_loader import load_text_from_directory
from model import Transformer
from nn_components.loss import SoftmaxCrossEntropy
from optimizer import Adam
from tokenizer import Tokenizer
from trainer import Trainer
from utils import load_checkpoint, save_checkpoint


def setup_logging(log_path: str):
    """Configures logging to file and console."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


def load_and_prepare_data(data_dir: str, tokenizer_path: str, validation_split: float):
    """Initializes the tokenizer and loads the training data."""
    logging.info("Initializing tokenizer and loading data...")
    tokenizer = Tokenizer(tokenizer_path)

    all_text = load_text_from_directory(data_dir)
    if not all_text:
        raise ValueError(f"Failed to load text from directory: {data_dir}")

    data_tokens = tokenizer.encode(all_text, add_special_tokens=True)
    split_idx = int(len(data_tokens) * (1 - validation_split))
    train_data, val_data = data_tokens[:split_idx], data_tokens[split_idx:]

    logging.info(f"Data loaded. Vocab size: {tokenizer.vocab_size}. "
                 f"Train tokens: {len(train_data)}. Validation tokens: {len(val_data)}.")
    return tokenizer, train_data, val_data


def initialize_components(config: Config, vocab_size: int):
    """Initializes the model, loss function, and optimizer."""
    logging.info("Initializing model, loss function, and optimizer...")
    model = Transformer(vocab_size=vocab_size, model_config=config.model, ltm_config=config.ltm)
    policy_loss_fn = SoftmaxCrossEntropy()

    optim_config = config.optimizer.model_dump()
    optim_config.pop('max_norm')
    optimizer = Adam(**optim_config)

    return model, policy_loss_fn, optimizer


def main():
    """Orchestrates the agent-centric training process."""
    parser = argparse.ArgumentParser(description="Agent-centric Transformer Training")
    parser.add_argument('--model-name', type=str, required=True,
                        help="Name of the model to train. A directory with this name will be created under 'models/'.")
    parser.add_argument('--resume-from', type=str,
                        help="Name of the model to resume training from. "
                             "Loads config and weights from 'models/<resume-from>/'.")
    args = parser.parse_args()

    model_dir = os.path.join('models', args.model_name)
    resume_dir = os.path.join('models', args.resume_from) if args.resume_from else None

    # --- Configuration and Setup ---
    if resume_dir:
        if not os.path.isdir(resume_dir):
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
        config_path = os.path.join(resume_dir, 'config.json')
        logging.info(f"Resuming training. Loading config from {config_path}")
    else:
        if os.path.exists(model_dir):
            raise FileExistsError(f"Model directory '{model_dir}' already exists. "
                                  "Choose a different name or use --resume-from.")
        os.makedirs(model_dir, exist_ok=True)
        config_path = 'config.json'
        # Copy the base config to the new model's directory
        shutil.copy(config_path, os.path.join(model_dir, 'config.json'))
        logging.info(f"Starting new training. Model directory created at {model_dir}")

    setup_logging(os.path.join(model_dir, 'training.log'))
    logging.info(f"--- Starting Training for Model: {args.model_name} ---")

    config = Config.from_json(config_path)
    set_backend(config.hardware.device)

    # --- Data and Component Initialization ---
    tokenizer, train_data, val_data = load_and_prepare_data(
        config.evolution.data_dir, config.evolution.data_dir, config.evolution.validation_split)
    model, loss_fn, optimizer = initialize_components(config, tokenizer.vocab_size)

    trainer = Trainer(config, model, optimizer, loss_fn, tokenizer, train_data, val_data)

    # --- State and Checkpoint Loading ---
    start_epoch, current_step, best_val_loss, epochs_no_improve = 0, 0, float('inf'), 0
    checkpoint_path = os.path.join(model_dir, 'checkpoint.npz')
    weights_path = os.path.join(model_dir, 'model.npz')
    best_model_path = os.path.join(model_dir, 'best_model.npz')

    # If resuming, load weights first, then checkpoint
    if resume_dir:
        resume_weights_path = os.path.join(resume_dir, 'model.npz')
        if os.path.exists(resume_weights_path):
            model.load_weights(resume_weights_path)
            logging.info(f"Loaded model weights from {resume_weights_path}")

    if os.path.exists(checkpoint_path):
        state, _ = load_checkpoint(model, optimizer, checkpoint_path)
        if state:
            start_epoch = state.get('epoch', 0)
            current_step = state.get('current_step', 0)
            best_val_loss = state.get('best_val_loss', float('inf'))
            epochs_no_improve = state.get('epochs_no_improve', 0)
            logging.info(f"Resuming from checkpoint. Epoch: {start_epoch}, Step: {current_step}.")

    # --- Training Loop ---
    pretrain_epochs = config.evolution.pretrain_epochs
    total_epochs = pretrain_epochs + config.evolution.evolution_epochs

    logging.info(f"Total epochs planned: {total_epochs} ({pretrain_epochs} pre-train, "
                 f"{config.evolution.evolution_epochs} evolution)")

    for epoch in range(start_epoch, total_epochs):
        is_pretrain = epoch < pretrain_epochs
        phase = "Pre-training" if is_pretrain else "Evolution"
        phase_epoch = epoch if is_pretrain else epoch - pretrain_epochs
        phase_total_epochs = pretrain_epochs if is_pretrain else config.evolution.evolution_epochs

        logging.info(f"--- Starting {phase} Epoch {phase_epoch + 1}/{phase_total_epochs} ---")

        if is_pretrain:
            avg_loss, epoch_time, current_step = trainer.train_pretrain_epoch(current_step)
        else:
            epoch_time = trainer.run_evolution_cycle()
            avg_loss = None # Not directly comparable in evolution phase

        val_loss = trainer.run_validation()
        log_msg = (f"Epoch {phase} {phase_epoch + 1}/{phase_total_epochs} | "
                   f"Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s")
        if avg_loss is not None:
            log_msg += f" | Avg Loss: {avg_loss:.4f}"
        if is_pretrain:
            log_msg += f" | LR: {optimizer.lr:.6f}"
        logging.info(log_msg)

        # --- Checkpointing and Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss, epochs_no_improve = val_loss, 0
            model.save_weights(best_model_path, config.model_dump())
            logging.info(f"New best model saved with Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in validation loss for {epochs_no_improve} epochs.")

        state = {'epoch': epoch + 1, 'current_step': current_step,
                 'best_val_loss': best_val_loss, 'epochs_no_improve': epochs_no_improve}
        save_checkpoint(model, optimizer, state, config.model_dump(), checkpoint_path)

        if epochs_no_improve >= config.evolution.early_stopping_patience:
            logging.warning(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
            break

    logging.info("Training complete!")
    model.save_weights(weights_path, config.model_dump())
    logging.info(f"Final model weights saved to {weights_path}")


if __name__ == "__main__":
    main()
