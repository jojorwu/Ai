"""
Main script for agent-centric training of the Transformer model using PyTorch.
"""
import argparse
import logging
import os
import shutil

import torch
from accelerate import Accelerator
from torch import nn
from torch.optim import Adam

from config import Config, TransformerConfig
from data_loader import load_multimodal_data_from_directory
from model import Transformer
from tokenizer import Tokenizer
from trainer import create_trainer
from utils import main_entrypoint, setup_logging


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


def setup_environment(args):
    """Sets up directories, logging, and configuration."""
    model_dir = os.path.join('models', args.model_name)
    resume_dir = os.path.join(
        'models', args.resume_from
    ) if args.resume_from else None

    if resume_dir:
        config_path = os.path.join(resume_dir, 'config.json')
        os.makedirs(model_dir, exist_ok=True)
        log_path = os.path.join(model_dir, 'training.log')
        setup_logging(log_path)
        logging.info(
            "Resuming training from '%s'. New logs in '%s'.",
            args.resume_from, args.model_name
        )
    else:
        if os.path.exists(model_dir):
            raise FileExistsError(f"Model directory '{model_dir}' already exists.")
        os.makedirs(model_dir)
        config_path = 'config.json'
        log_path = os.path.join(model_dir, 'training.log')
        setup_logging(log_path)
        shutil.copy(config_path, os.path.join(model_dir, 'config.json'))
        logging.info("Starting new training run: '%s'.", args.model_name)

    config = Config.from_json(config_path)
    return config, model_dir, resume_dir

def run_training_loop(trainer, config, model_dir):
    """Executes the main training loop."""
    best_val_loss = float('inf')
    epochs_no_improve = 0
    total_epochs = (
        config.evolution.pretrain_epochs + config.evolution.evolution_epochs
    )
    model = trainer.get_model()

    for epoch in range(total_epochs):
        is_pretrain = epoch < config.evolution.pretrain_epochs
        phase = "Pre-training" if is_pretrain else "Evolution"
        phase_epoch = epoch if is_pretrain else epoch - config.evolution.pretrain_epochs
        total_phase_epochs = (
            config.evolution.pretrain_epochs
            if is_pretrain
            else config.evolution.evolution_epochs
        )
        logging.info(
            "\n--- %s Epoch %d/%d ---", phase, phase_epoch + 1, total_phase_epochs
        )

        if is_pretrain:
            avg_loss, epoch_time = trainer.train_pretrain_epoch()
            logging.info("    - Average Loss: %.4f", avg_loss)
        else:
            epoch_time = trainer.run_evolution_cycle()

        val_loss = trainer.run_validation()
        logging.info(
            "    - Validation Loss: %.4f, Epoch Time: %.2fs", val_loss, epoch_time
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(
                model.state_dict(), os.path.join(model_dir, 'best_model.pt')
            )
            logging.info(
                "    - New best model saved (Val Loss: %.4f)", best_val_loss
            )
        else:
            epochs_no_improve += 1
            logging.info(
                "    - No improvement in validation loss for %d epochs.",
                epochs_no_improve
            )

        if epochs_no_improve >= config.evolution.early_stopping_patience:
            logging.warning("Early stopping triggered.")
            break

@main_entrypoint
def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Agent-centric Transformer Training with PyTorch."
    )
    parser.add_argument(
        '--model-name', type=str, required=True, help="Name for the model."
    )
    parser.add_argument(
        '--resume-from', type=str, help="Resume training from an existing model."
    )
    parser.add_argument(
        '--load-in-4bit', action='store_true', help="Load the model in 4-bit."
    )
    args = parser.parse_args()

    config, model_dir, resume_dir = setup_environment(args)
    accelerator = Accelerator()
    tokenizer, train_data, val_data = load_and_prepare_data(
        config.evolution.data_dir,
        config.evolution.data_dir,
        config.evolution.validation_split,
    )
    if not resume_dir:
        shutil.copy(
            os.path.join(config.evolution.data_dir, 'tokenizer_vocab.json'),
            model_dir,
        )

    trainer = create_trainer(
        config, train_data, val_data, tokenizer, accelerator, args.load_in_4bit
    )
    model = trainer.get_model()

    if resume_dir:
        weights_path = os.path.join(resume_dir, 'best_model.pt')
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path))
            logging.info("Loaded model weights from %s", weights_path)

    run_training_loop(trainer, config, model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
    logging.info(
        "\nTraining complete! Final model saved to: %s", model_dir
    )

if __name__ == "__main__":
    main()
