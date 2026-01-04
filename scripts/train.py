"""
Main script for agent-centric training of the Transformer model using PyTorch.
"""
import json
import logging
import os
import shutil

import torch
from accelerate import Accelerator

from src.config import TrainConfig
from src.data.data_loader import load_multimodal_data_from_directory
from src.data.tokenizer import Tokenizer
from src.trainer import create_trainer, DataComponents
from src.utils.cli import create_main_parser
from src.utils.core import main_entrypoint, setup_logging


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
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    resume_from_checkpoint = (
        os.path.join('models', args.resume_from, 'checkpoints')
        if args.resume_from
        else None
    )

    if resume_from_checkpoint:
        config_path = os.path.join('models', args.resume_from, 'config.json')
        os.makedirs(model_dir, exist_ok=True)
        log_path = os.path.join(model_dir, 'training.log')
        setup_logging(log_path)
        logging.info(
            "Resuming training from checkpoint '%s'. New model and logs will be in '%s'.",
            resume_from_checkpoint,
            args.model_name,
        )
    else:
        if os.path.exists(model_dir):
            raise FileExistsError(f"Model directory '{model_dir}' already exists.")
        os.makedirs(model_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        config_path = 'config_train.json'
        log_path = os.path.join(model_dir, 'training.log')
        setup_logging(log_path)
        logging.info("Starting new training run: '%s'.", args.model_name)

    config = TrainConfig.from_json(config_path)
    return config, model_dir, checkpoint_dir, resume_from_checkpoint


def save_updated_config(config: TrainConfig, tokenizer: Tokenizer, model_dir: str):
    """
    Saves a final, generation-ready config by merging training and generation settings.
    """
    # Load the base generation config to get the 'generation' section
    with open("config_generate.json", "r", encoding="utf-8") as f:
        gen_config_data = json.load(f)

    # Create a serializable dictionary from the Pydantic model
    final_config = config.model_dump()
    final_config['generation'] = gen_config_data.get('generation')
    final_config['dynamic_parameters'] = gen_config_data.get('dynamic_parameters')
    final_config['model']['vocab_size'] = tokenizer.vocab_size

    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(final_config, f, indent=4)


class TrainingState:
    """A simple class to hold and manage the training state like epoch."""
    def __init__(self, epoch=0):
        self.epoch = epoch

    def state_dict(self):
        """Returns the state of the training."""
        return {'epoch': self.epoch}

    def load_state_dict(self, state_dict):
        """Loads the training state."""
        self.epoch = state_dict['epoch']

def run_training_loop(trainer, config, checkpoint_dir, accelerator, args): # pylint: disable=too-many-locals
    """Executes the main training loop."""
    if accelerator.is_main_process:
        logging.info("Starting training loop...")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    total_epochs = (
        config.evolution.pretrain_epochs + config.evolution.evolution_epochs
    )
    # This state will be managed by Accelerator
    training_state = TrainingState()
    accelerator.register_for_checkpointing(training_state)

    if accelerator.is_main_process and args.resume_from:
        try:
            checkpoint_path = os.path.join('models', args.resume_from, 'checkpoints')
            accelerator.load_state(checkpoint_path)
            logging.info(
                "Successfully loaded checkpoint. Starting from epoch %d.",
                training_state.epoch
            )
        except FileNotFoundError:
            logging.warning("Checkpoint not found at the specified path. Starting from scratch.")


    start_epoch = training_state.epoch
    for epoch in range(start_epoch, total_epochs):
        training_state.epoch = epoch
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
            if accelerator.is_main_process:
                accelerator.log({
                    "avg_loss": avg_loss,
                    "epoch_time": epoch_time,
                    "learning_rate": trainer.get_learning_rate()
                }, step=epoch)
            logging.info("    - Average Loss: %.4f", avg_loss)
        else:
            epoch_time = trainer.run_evolution_cycle()
            if accelerator.is_main_process:
                accelerator.log({"epoch_time": epoch_time}, step=epoch)

        val_loss = trainer.run_validation()
        if accelerator.is_main_process:
            accelerator.log({"val_loss": val_loss}, step=epoch)
        logging.info(
            "    - Validation Loss: %.4f, Epoch Time: %.2fs", val_loss, epoch_time
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            accelerator.save_state(checkpoint_dir)
            logging.info(
                "    - New best checkpoint saved (Val Loss: %.4f)", best_val_loss
            )
        else:
            epochs_no_improve += 1
            logging.info(
                "    - No improvement in validation loss for %d epochs.",
                epochs_no_improve,
            )

        if epochs_no_improve >= config.evolution.early_stopping_patience:
            logging.warning("Early stopping triggered.")
            break

@main_entrypoint
def main():
    """Main training script."""
    parser = create_main_parser()
    parser.add_argument(
        '--resume-from', type=str, help="Name of the model run to resume from."
    )
    parser.add_argument(
        '--wandb', action='store_true', help="Enable Weights & Biases logging."
    )
    # Require model-name for training
    for action in parser._actions:  # pylint: disable=protected-access
        if action.dest == 'model_name':
            action.required = True
            break
    args = parser.parse_args()

    config, model_dir, checkpoint_dir, resume_from_checkpoint = setup_environment(args)
    if args.hardware_strategy:
        config.hardware.strategy = args.hardware_strategy
        logging.info(
            "Overriding hardware strategy with '%s'", args.hardware_strategy
        )

    accelerator = Accelerator(mixed_precision="fp16", log_with="wandb" if args.wandb else None)
    if accelerator.is_main_process and args.wandb:
        accelerator.init_trackers(
            project_name="transformer-project",
            config=config.model_dump()
        )

    tokenizer, train_data, val_data = load_and_prepare_data(
        config.evolution.data_dir,
        config.evolution.data_dir,
        config.evolution.validation_split,
    )

    # Save config and tokenizer vocab only on new runs
    if not resume_from_checkpoint:
        save_updated_config(config, tokenizer, model_dir)
        tokenizer.save_vocab(model_dir)

    data_components = DataComponents(
        tokenizer=tokenizer, train_data=train_data, val_data=val_data
    )
    trainer = create_trainer(
        config, data_components, accelerator, args.load_in_4bit
    )

    run_training_loop(trainer, config, checkpoint_dir, accelerator, args)

    # Save the final, unwrapped model for easy inference
    unwrapped_model = accelerator.unwrap_model(trainer.get_model())
    torch.save(unwrapped_model.state_dict(), os.path.join(model_dir, 'model.pt'))
    logging.info(
        "\nTraining complete! Final model saved to: %s", model_dir
    )

if __name__ == "__main__":
    main()
