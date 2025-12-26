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
    # Load multimodal data and concatenate text for pre-training
    multimodal_data = load_multimodal_data_from_directory(data_dir)
    if not multimodal_data:
        raise ValueError(f"Failed to load any data from directory: {data_dir}")
    all_text = " ".join([text for text, _ in multimodal_data])
    data_tokens = tokenizer.encode(all_text, add_special_tokens=True)
    split_idx = int(len(data_tokens) * (1 - validation_split))
    train_data, val_data = data_tokens[:split_idx], data_tokens[split_idx:]
    logging.info(f"Data loaded successfully. Vocab size: {tokenizer.vocab_size}, "
                 f"Train tokens: {len(train_data)}, Validation tokens: {len(val_data)}.")
    return tokenizer, train_data, val_data


def initialize_components(config: Config, vocab_size: int, tokenizer):
    """Initializes the model, loss function, and optimizer."""
    logging.info("Initializing model, loss, and optimizer...")
    model = Transformer(vocab_size=vocab_size, model_config=config.model,
                        vision_config=config.vision, ltm_config=config.ltm, tokenizer=tokenizer)
    policy_loss_fn = SoftmaxCrossEntropy()
    optimizer = Adam(config.optimizer)
    logging.info("Model and optimizer initialized.")
    return model, policy_loss_fn, optimizer


def _parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Agent-centric Transformer Training")
    parser.add_argument(
        '--model-name', type=str, required=True,
        help="Name for the new model. A directory will be created under 'models/'.")
    parser.add_argument(
        '--resume-from', type=str,
        help="Name of an existing model to resume training from.")
    return parser.parse_args()


def _setup_environment(args):
    """Sets up the training environment, directories, and logging."""
    model_dir = os.path.join('models', args.model_name)
    resume_dir = os.path.join('models', args.resume_from) if args.resume_from else None

    if resume_dir:
        if not os.path.isdir(resume_dir):
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
        config_path = os.path.join(resume_dir, 'config.json')
        log_file_path = os.path.join(model_dir, 'training.log')
        os.makedirs(model_dir, exist_ok=True)
        setup_logging(log_file_path)
        logging.info("Resuming training from '%s'. New logs will be saved to '%s'.",
                     args.resume_from, args.model_name)
    else:
        if os.path.exists(model_dir):
            raise FileExistsError(f"Model directory '{model_dir}' already exists. "
                                  "Choose a different name or use --resume-from.")
        os.makedirs(model_dir)
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


def _initialize_training_state(model, optimizer, model_dir, resume_dir):
    """Initializes or resumes the training state."""
    start_epoch, current_step, best_val_loss, epochs_no_improve = 0, 0, float('inf'), 0
    checkpoint_path = os.path.join(model_dir, 'checkpoint.npz')

    if resume_dir:
        weights_path = os.path.join(resume_dir, 'best_model.npz')
        if not os.path.exists(weights_path):
            weights_path = os.path.join(resume_dir, 'model.npz')
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            logging.info("Loaded model weights from %s", weights_path)

    if os.path.exists(checkpoint_path):
        state, _ = load_checkpoint(model, optimizer, checkpoint_path)
        if state:
            start_epoch = state.get('epoch', 0)
            current_step = state.get('current_step', 0)
            best_val_loss = state.get('best_val_loss', float('inf'))
            epochs_no_improve = state.get('epochs_no_improve', 0)
            logging.info("Resuming from checkpoint. Start Epoch: %d, Step: %d.", start_epoch, current_step)
    return start_epoch, current_step, best_val_loss, epochs_no_improve


def _run_epoch(epoch, config, trainer, optimizer):
    """Runs a single epoch of training (either pre-training or evolution)."""
    is_pretrain = epoch < config.evolution.pretrain_epochs
    phase = "Pre-training" if is_pretrain else "Evolution"
    phase_epoch = epoch if is_pretrain else epoch - \
                  config.evolution.pretrain_epochs
    total_phase_epochs = config.evolution.pretrain_epochs if is_pretrain else \
        config.evolution.evolution_epochs

    logging.info("\n--- %s Epoch %d/%d ---", phase,
                 phase_epoch + 1, total_phase_epochs)

    avg_loss, epoch_time, current_step_delta = None, 0, 0
    if is_pretrain:
        avg_loss, epoch_time, current_step_delta = trainer.train_pretrain_epoch(0)
    else:
        epoch_time = trainer.run_evolution_cycle()

    val_loss = trainer.run_validation()
    log_parts = [f"Validation Loss: {val_loss:.4f}", f"Epoch Time: {epoch_time:.2f}s"]
    if avg_loss is not None:
        log_parts.insert(0, f"Average Loss: {avg_loss:.4f}")
    if is_pretrain:
        log_parts.append(f"Learning Rate: {optimizer.lr:.6f}")
    logging.info("    - " + "\n    - ".join(log_parts))
    return val_loss, current_step_delta


def _handle_epoch_end(val_loss, best_val_loss, epochs_no_improve, model, config, model_dir):
    """Handles end-of-epoch logic like saving the best model."""
    if val_loss < best_val_loss:
        best_val_loss, epochs_no_improve = val_loss, 0
        model.save_weights(os.path.join(model_dir, 'best_model.npz'), config.model_dump())
        logging.info("    - New best model saved (Val Loss: %.4f)", best_val_loss)
    else:
        epochs_no_improve += 1
        logging.info("    - No improvement in validation loss for %d epochs.", epochs_no_improve)
    return best_val_loss, epochs_no_improve


def _run_training_loop(trainer, config, model, optimizer, model_dir, resume_dir):
    """Executes the main training loop."""
    start_epoch, current_step, best_val_loss, epochs_no_improve = _initialize_training_state(
        model, optimizer, model_dir, resume_dir)
    total_epochs = config.evolution.pretrain_epochs + config.evolution.evolution_epochs
    logging.info("Starting training loop for %d total epochs.", total_epochs)

    for epoch in range(start_epoch, total_epochs):
        val_loss, current_step_delta = _run_epoch(epoch, config, trainer, optimizer)
        current_step += current_step_delta
        best_val_loss, epochs_no_improve = _handle_epoch_end(
            val_loss, best_val_loss, epochs_no_improve, model, config, model_dir)
        state = {'epoch': epoch + 1, 'current_step': current_step,
                 'best_val_loss': best_val_loss,
                 'epochs_no_improve': epochs_no_improve}
        save_checkpoint(model, optimizer, state, config.model_dump(),
                        os.path.join(model_dir, 'checkpoint.npz'))
        if epochs_no_improve >= config.evolution.early_stopping_patience:
            logging.warning("Early stopping triggered. Ending training.")
            break


def main():
    """Orchestrates the agent-centric training process."""
    args = _parse_args()
    config, model_dir, resume_dir = _setup_environment(args)

    tokenizer, train_data, val_data = load_and_prepare_data(
        config.evolution.data_dir, config.evolution.data_dir, config.evolution.validation_split)

    if not resume_dir:
        shutil.copy(os.path.join(config.evolution.data_dir, 'tokenizer_vocab.json'),
                    os.path.join(model_dir, 'tokenizer_vocab.json'))
        logging.info("Copied tokenizer vocabulary to %s", model_dir)

    model, loss_fn, optimizer = initialize_components(config, tokenizer.vocab_size, tokenizer)
    total_params = model.count_parameters()
    logging.info("Model initialized with %d trainable parameters.", total_params)

    trainer = Trainer(
        training_components={'model': model, 'optimizer': optimizer,
                             'loss_fn': loss_fn, 'tokenizer': tokenizer},
        training_data={'train': train_data, 'validation': val_data},
        config=config
    )

    _run_training_loop(trainer, config, model, optimizer, model_dir, resume_dir)

    final_weights_path = os.path.join(model_dir, 'model.npz')
    model.save_weights(final_weights_path, config.model_dump())
    logging.info("\n--------------------------------------------------")
    logging.info("Training complete!")
    logging.info("    - Total Parameters: %d", total_params)
    logging.info("    - Final model saved to: %s", final_weights_path)
    logging.info("--------------------------------------------------")

if __name__ == "__main__":
    main()
