"""
Main script for agent-centric training of the Transformer model.
"""
import argparse
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Tuple, Optional

from backend import set_backend
from config import Config
from data_loader import load_multimodal_data_from_directory
from model import Transformer
from nn_components.loss import SoftmaxCrossEntropy
from optimizer import Adam
from tokenizer import Tokenizer
from trainer import Trainer
from utils import load_checkpoint


@dataclass
class TrainingState:
    """Encapsulates the state of a training run."""
    epoch: int = 0
    current_step: int = 0
    best_val_loss: float = float('inf')
    epochs_no_improve: int = 0


@dataclass
class TrainingData:
    """Encapsulates all data required for training."""
    tokenizer: Tokenizer
    train_data: list
    val_data: list


@dataclass
class TrainingComponents:
    """Encapsulates the core components for training."""
    model: Transformer
    optimizer: Adam
    loss_fn: SoftmaxCrossEntropy


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


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Agent-centric Transformer Training")
    parser.add_argument('--model-name', type=str, required=True,
                        help="Name for the new model. A directory will be created under 'models/'.")
    parser.add_argument('--resume-from', type=str,
                        help="Name of an existing model to resume training from.")
    return parser.parse_args()


def setup_training_environment(args: argparse.Namespace) -> Tuple[str, Optional[str], Config]:
    """Sets up directories, logging, and loads the configuration."""
    model_dir = os.path.join('models', args.model_name)
    resume_dir = os.path.join('models', args.resume_from) if args.resume_from else None

    if resume_dir:
        if not os.path.isdir(resume_dir):
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
        config_path = os.path.join(resume_dir, 'config.json')
        os.makedirs(model_dir, exist_ok=True)
        setup_logging(os.path.join(model_dir, 'training.log'))
        logging.info("Resuming training from '%s'. New logs saved to '%s'.",
                     args.resume_from, args.model_name)
        logging.info("Loading config from %s", config_path)
    else:
        if os.path.exists(model_dir):
            raise FileExistsError(f"Model directory '{model_dir}' already exists. "
                                  "Choose a different name or use --resume-from.")
        os.makedirs(model_dir)
        config_path = 'config.json'
        setup_logging(os.path.join(model_dir, 'training.log'))
        shutil.copy(config_path, os.path.join(model_dir, 'config.json'))
        logging.info("Starting new training run: '%s'.", args.model_name)

    logging.info("--------------------------------------------------")
    logging.info("Model: %s", args.model_name)
    logging.info("--------------------------------------------------")

    config = Config.from_json(config_path)
    set_backend(config.hardware.device)
    return model_dir, resume_dir, config


def load_and_prepare_data(config: Config) -> TrainingData:
    """Initializes the tokenizer and loads the training data."""
    logging.info("Initializing tokenizer and loading data...")
    tokenizer = Tokenizer(config.evolution.data_dir)
    text_data, _ = load_multimodal_data_from_directory(config.evolution.data_dir)

    if not text_data:
        raise ValueError(f"Failed to load text from directory: {config.evolution.data_dir}")

    data_tokens = tokenizer.encode(text_data, add_special_tokens=True)
    split_idx = int(len(data_tokens) * (1 - config.evolution.validation_split))

    train_tokens, val_tokens = data_tokens[:split_idx], data_tokens[split_idx:]

    logging.info("Data loaded. Vocab size: %d, Train tokens: %d, Val tokens: %d",
                 tokenizer.vocab_size, len(train_tokens), len(val_tokens))
    return TrainingData(tokenizer=tokenizer, train_data=train_tokens, val_data=val_tokens)


def initialize_components(config: Config, vocab_size: int, tokenizer: Tokenizer) -> TrainingComponents:
    """Initializes the model, loss function, and optimizer."""
    logging.info("Initializing model, loss, and optimizer...")
    model = Transformer(vocab_size=vocab_size, model_config=config.model,
                        vision_config=config.vision, ltm_config=config.ltm, tokenizer=tokenizer)
    loss_fn = SoftmaxCrossEntropy()
    optimizer = Adam(config.optimizer)
    logging.info("Model and optimizer initialized.")
    return TrainingComponents(model=model, optimizer=optimizer, loss_fn=loss_fn)


def load_training_state(resume_dir: Optional[str], model_dir: str, model: Transformer,
                        optimizer: Adam) -> TrainingState:
    """Loads model weights and optimizer state from a checkpoint if available."""
    state = TrainingState()
    checkpoint_path = os.path.join(model_dir, 'checkpoint.npz')

    if resume_dir:
        resume_weights_path = os.path.join(resume_dir, 'best_model.npz')
        if not os.path.exists(resume_weights_path):
            resume_weights_path = os.path.join(resume_dir, 'model.npz')
        if os.path.exists(resume_weights_path):
            model.load_weights(resume_weights_path)
            logging.info("Loaded model weights from %s", resume_weights_path)

    if os.path.exists(checkpoint_path):
        loaded_state, _ = load_checkpoint(model, optimizer, checkpoint_path)
        if loaded_state:
            state = TrainingState(**loaded_state)
            logging.info("Resuming from checkpoint. Start Epoch: %d, Step: %d.",
                         state.epoch, state.current_step)
    return state


def main():
    """Orchestrates the agent-centric training process."""
    args = parse_arguments()
    model_dir, resume_dir, config = setup_training_environment(args)

    training_data = load_and_prepare_data(config)
    components = initialize_components(config, training_data.tokenizer.vocab_size,
                                       training_data.tokenizer)

    total_params = components.model.count_parameters()
    logging.info("Model initialized with %s trainable parameters.", f'{total_params:,}')

    state = load_training_state(resume_dir, model_dir, components.model, components.optimizer)

    trainer = Trainer(config, components, training_data)
    trainer.run_training(model_dir, state)

    final_weights_path = os.path.join(model_dir, 'model.npz')
    components.model.save_weights(final_weights_path, config.model_dump())
    logging.info("\n--------------------------------------------------")
    logging.info("Training complete!")
    logging.info("    - Total Parameters: %s", f'{total_params:,}')
    logging.info("    - Final model saved to: %s", final_weights_path)
    logging.info("--------------------------------------------------")


if __name__ == "__main__":
    main()
