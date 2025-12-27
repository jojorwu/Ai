"""
Main script for agent-centric training of the Transformer model using PyTorch.
"""
import argparse
import logging
import os
import shutil
import torch
import torch.nn as nn
from torch.optim import Adam
from accelerate import Accelerator

from config import Config, TransformerConfig
from data_loader import load_multimodal_data_from_directory
from model import Transformer
from tokenizer import Tokenizer
from trainer import Trainer

def setup_logging(log_path: str):
    """Configures logging to file and console."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    logging.getLogger().setLevel(logging.INFO)

def load_and_prepare_data(data_dir: str, tokenizer_path: str, validation_split: float):
    """Initializes tokenizer and loads data."""
    tokenizer = Tokenizer(tokenizer_path)
    multimodal_data = load_multimodal_data_from_directory(data_dir)
    if not multimodal_data:
        raise ValueError(f"No data found in {data_dir}")
    all_text = " ".join([text for text, _ in multimodal_data])
    data_tokens = tokenizer.encode(all_text, add_special_tokens=True)
    split_idx = int(len(data_tokens) * (1 - validation_split))
    return tokenizer, data_tokens[:split_idx], data_tokens[split_idx:]

def initialize_components(config: Config, vocab_size: int, tokenizer, load_in_4bit: bool):
    """Initializes the PyTorch model, loss functions, and optimizer."""
    transformer_config = TransformerConfig(
        vocab_size=vocab_size, model=config.model, vision=config.vision,
        ltm=config.ltm, tokenizer=tokenizer
    )
    model = Transformer(transformer_config, load_in_4bit=load_in_4bit)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.optimizer.learning_rate)
    return model, policy_loss_fn, value_loss_fn, optimizer

def setup_environment(args):
    """Sets up directories, logging, and configuration."""
    model_dir = os.path.join('models', args.model_name)
    resume_dir = os.path.join('models', args.resume_from) if args.resume_from else None

    if resume_dir:
        config_path = os.path.join(resume_dir, 'config.json')
        os.makedirs(model_dir, exist_ok=True)
        log_path = os.path.join(model_dir, 'training.log')
        setup_logging(log_path)
        logging.info(f"Resuming training from '{args.resume_from}'. New logs in '{args.model_name}'.")
    else:
        if os.path.exists(model_dir):
            raise FileExistsError(f"Model directory '{model_dir}' already exists.")
        os.makedirs(model_dir)
        config_path = 'config.json'
        log_path = os.path.join(model_dir, 'training.log')
        setup_logging(log_path)
        shutil.copy(config_path, os.path.join(model_dir, 'config.json'))
        logging.info(f"Starting new training run: '{args.model_name}'.")

    config = Config.from_json(config_path)
    return config, model_dir, resume_dir

def run_training_loop(trainer, config, model, model_dir):
    """Executes the main training loop."""
    best_val_loss = float('inf')
    epochs_no_improve = 0
    total_epochs = config.evolution.pretrain_epochs + config.evolution.evolution_epochs

    for epoch in range(total_epochs):
        is_pretrain = epoch < config.evolution.pretrain_epochs
        phase = "Pre-training" if is_pretrain else "Evolution"
        phase_epoch = epoch if is_pretrain else epoch - config.evolution.pretrain_epochs
        total_phase_epochs = config.evolution.pretrain_epochs if is_pretrain else config.evolution.evolution_epochs

        logging.info(f"\n--- {phase} Epoch {phase_epoch + 1}/{total_phase_epochs} ---")

        if is_pretrain:
            avg_loss, epoch_time = trainer.train_pretrain_epoch()
            logging.info(f"    - Average Loss: {avg_loss:.4f}")
        else:
            epoch_time = trainer.run_evolution_cycle()

        val_loss = trainer.run_validation()
        logging.info(f"    - Validation Loss: {val_loss:.4f}, Epoch Time: {epoch_time:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))
            logging.info(f"    - New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            logging.info(f"    - No improvement in validation loss for {epochs_no_improve} epochs.")

        if epochs_no_improve >= config.evolution.early_stopping_patience:
            logging.warning("Early stopping triggered.")
            break

def main():
    parser = argparse.ArgumentParser(description="Agent-centric Transformer Training with PyTorch.")
    parser.add_argument('--model-name', type=str, required=True, help="Name for the model.")
    parser.add_argument('--resume-from', type=str, help="Resume training from an existing model.")
    parser.add_argument('--load-in-4bit', action='store_true', help="Load the model in 4-bit.")
    args = parser.parse_args()

    try:
        config, model_dir, resume_dir = setup_environment(args)
        accelerator = Accelerator()

        tokenizer, train_data, val_data = load_and_prepare_data(
            config.evolution.data_dir, config.evolution.data_dir, config.evolution.validation_split
        )
        if not resume_dir:
            shutil.copy(os.path.join(config.evolution.data_dir, 'tokenizer_vocab.json'), model_dir)

        model, policy_loss, value_loss, opt = initialize_components(config, tokenizer.vocab_size, tokenizer, args.load_in_4bit)

        if resume_dir:
            weights_path = os.path.join(resume_dir, 'best_model.pt')
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path))
                logging.info(f"Loaded model weights from {weights_path}")

        model, opt, policy_loss, value_loss = accelerator.prepare(model, opt, policy_loss, value_loss)

        trainer = Trainer(
            model=model, optimizer=opt, policy_loss_fn=policy_loss, value_loss_fn=value_loss,
            tokenizer=tokenizer, train_data=train_data, val_data=val_data, config=config,
            accelerator=accelerator
        )

        run_training_loop(trainer, config, model, model_dir)

        torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
        logging.info(f"\nTraining complete! Final model saved to: {model_dir}")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
