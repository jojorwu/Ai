"""
Main script for agent-centric training of the Transformer model.
"""
import argparse
import logging
import os
import shutil

from backend import set_backend
from config import Config
from data_loader import load_multimodal_data_from_directory
from model import Transformer
from nn_components.loss import SoftmaxCrossEntropy
from optimizer import Adam
from tokenizer import Tokenizer
from trainer import Trainer
from utils import load_checkpoint, save_checkpoint


def setup_logging(log_path: str):
    """Configures logging to file and console with improved formatting."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.getLogger().setLevel(logging.INFO)


def load_and_prepare_data(data_dir: str, tokenizer_path: str, validation_split: float):
    """Initializes the tokenizer and loads the training data."""
    logging.info("Initializing tokenizer and loading data...")
    tokenizer = Tokenizer(tokenizer_path)
    multimodal_data = load_multimodal_data_from_directory(data_dir)
    if not multimodal_data:
        raise ValueError(f"Failed to load any data from directory: {data_dir}")

    all_text = " ".join([text for text, _ in multimodal_data if text])
    if not all_text.strip():
        raise ValueError(f"No text content found in data from directory: {data_dir}")

    data_tokens = tokenizer.encode(all_text, add_special_tokens=True)
    split_idx = int(len(data_tokens) * (1 - validation_split))
    train_data, val_data = data_tokens[:split_idx], data_tokens[split_idx:]
    logging.info("Data loaded. Vocab size: %d, Train tokens: %d, Val tokens: %d.",
                 tokenizer.vocab_size, len(train_data), len(val_data))
    return tokenizer, train_data, val_data


def initialize_components(config: Config, vocab_size: int, tokenizer):
    """Initializes the model, loss function, and optimizer."""
    logging.info("Initializing model, loss, and optimizer...")
    model = Transformer(vocab_size=vocab_size, model_config=config.model,
                        vision_config=config.vision, ltm_config=config.ltm,
                        tokenizer=tokenizer)
    policy_loss_fn = SoftmaxCrossEntropy()
    optim_config = config.optimizer.model_dump()
    optim_config.pop('max_norm')
    optimizer = Adam(**optim_config)
    logging.info("Model and optimizer initialized.")
    return model, policy_loss_fn, optimizer


def setup_training_environment(args):
    """Handles command-line arguments, sets up directories, logging, and config."""
    model_dir = os.path.join('models', args.model_name)
    resume_dir = os.path.join('models', args.resume_from) if args.resume_from else None

    # --- Configuration and Setup ---
    if resume_dir:
        if not os.path.isdir(resume_dir):
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
        config_path = os.path.join(resume_dir, 'config.json')
        log_file_path = os.path.join(model_dir, 'training.log')
        os.makedirs(model_dir, exist_ok=True)
        setup_logging(log_file_path)
        logging.info("Resuming training from '%s'. New logs in '%s'.",
                     args.resume_from, args.model_name)
        logging.info("Loading config from %s", config_path)
    else:
        if os.path.exists(model_dir):
            raise FileExistsError(f"Model directory '{model_dir}' already exists. "
                                  "Choose a different name or use --resume-from.")
        os.makedirs(model_dir, exist_ok=True)
        config_path = 'config.json'
        log_file_path = os.path.join(model_dir, 'training.log')
        setup_logging(log_file_path)
        shutil.copy(config_path, os.path.join(model_dir, 'config.json'))
        logging.info("Starting new training run: '%s'.", args.model_name)

    logging.info("--------------------------------------------------")
    logging.info("Model: %s", args.model_name)
    logging.info("Log file: %s", log_file_path)
    logging.info("--------------------------------------------------")

    config = Config.from_json(config_path)
    set_backend(config.hardware.device)
    return config, model_dir, resume_dir


def load_training_state(model, optimizer, tokenizer, config, model_dir, resume_dir):
    """Loads model weights and checkpoint state if resuming training."""
    state = {'epoch': 0, 'current_step': 0, 'best_val_loss': float('inf'), 'epochs_no_improve': 0}
    checkpoint_path = os.path.join(model_dir, 'checkpoint.npz')

    if resume_dir:
        weights_path = os.path.join(resume_dir, 'best_model.npz')
        if not os.path.exists(weights_path):
            weights_path = os.path.join(resume_dir, 'model.npz')

        if os.path.exists(weights_path):
            model = Transformer.load_model(weights_path, tokenizer.vocab_size, config, tokenizer)
            logging.info("Loaded model weights from %s", weights_path)

    if os.path.exists(checkpoint_path):
        loaded_state, _ = load_checkpoint(model, optimizer, checkpoint_path)
        if loaded_state:
            state.update(loaded_state)
            logging.info("Resuming from checkpoint. Start Epoch: %d, Step: %d.",
                         state['epoch'], state['current_step'])

    return model, state


def run_training_loop(trainer, model, optimizer, config, model_dir,
                      checkpoint_path, training_state):
    """The main training loop, including pre-training and evolution phases."""
    start_epoch = training_state['start_epoch']
    current_step = training_state['current_step']
    best_val_loss = training_state['best_val_loss']
    epochs_no_improve = training_state['epochs_no_improve']

    pretrain_epochs = config.evolution.pretrain_epochs
    total_epochs = pretrain_epochs + config.evolution.evolution_epochs
    logging.info("Starting training loop for %d total epochs.", total_epochs)

    for epoch in range(start_epoch, total_epochs):
        is_pretrain = epoch < pretrain_epochs
        phase = "Pre-training" if is_pretrain else "Evolution"
        phase_epoch = epoch if is_pretrain else epoch - pretrain_epochs
        phase_total_epochs = pretrain_epochs if is_pretrain else config.evolution.evolution_epochs

        logging.info("\n--- %s Epoch %d/%d ---", phase, phase_epoch + 1, phase_total_epochs)

        avg_loss, epoch_time = None, 0
        if is_pretrain:
            avg_loss, epoch_time, current_step = trainer.train_pretrain_epoch(current_step)
        else:
            epoch_time = trainer.run_evolution_cycle()

        val_loss = trainer.run_validation()
        log_msg = f"    - Validation Loss: {val_loss:.4f}\n    - Epoch Time: {epoch_time:.2f}s"
        if avg_loss is not None:
            log_msg = f"    - Average Loss: {avg_loss:.4f}\n{log_msg}"
        if is_pretrain:
            log_msg += f"\n    - Learning Rate: {optimizer.lr:.6f}"
        logging.info(log_msg)

        if val_loss < best_val_loss:
            best_val_loss, epochs_no_improve = val_loss, 0
            best_model_path = os.path.join(model_dir, 'best_model.npz')
            model.save_weights(best_model_path, config.model_dump())
            logging.info("    - New best model saved (Val Loss: %.4f)", best_val_loss)
        else:
            epochs_no_improve += 1
            logging.info("    - No improvement in validation loss for %d epochs.", epochs_no_improve)

        state = {'epoch': epoch + 1, 'current_step': current_step,
                 'best_val_loss': best_val_loss, 'epochs_no_improve': epochs_no_improve}
        save_checkpoint(model, optimizer, state, config.model_dump(), checkpoint_path)

        if epochs_no_improve >= config.evolution.early_stopping_patience:
            logging.warning("Early stopping triggered. Ending training.")
            break


def main():
    """Orchestrates the agent-centric training process."""
    parser = argparse.ArgumentParser(description="Agent-centric Transformer Training")
    parser.add_argument(
        '--model-name', type=str, required=True,
        help="Name for the new model. Directory created under 'models/'.")
    parser.add_argument(
        '--resume-from', type=str,
        help="Name of an existing model to resume training from.")
    args = parser.parse_args()

    config, model_dir, resume_dir = setup_training_environment(args)

    tokenizer, train_data, val_data = load_and_prepare_data(
        config.evolution.data_dir, config.evolution.data_dir,
        config.evolution.validation_split)

    model, loss_fn, optimizer = initialize_components(
        config, tokenizer.vocab_size, tokenizer)

    model, training_state = load_training_state(
        model, optimizer, tokenizer, config, model_dir, resume_dir)

    total_params = model.count_parameters()
    logging.info("Model initialized with %s trainable parameters.", f"{total_params:,}")

    trainer = Trainer(config, model, optimizer, loss_fn, tokenizer, train_data, val_data)
    checkpoint_path = os.path.join(model_dir, 'checkpoint.npz')

    run_training_loop(trainer, model, optimizer, config, model_dir,
                      checkpoint_path, training_state)

    final_weights_path = os.path.join(model_dir, 'model.npz')
    model.save_weights(final_weights_path, config.model_dump())
    logging.info("\n--------------------------------------------------")
    logging.info("Training complete!")
    logging.info("    - Total Parameters: %s", f"{total_params:,}")
    logging.info("    - Final model saved to: %s", final_weights_path)
    logging.info("--------------------------------------------------")

if __name__ == "__main__":
    main()
