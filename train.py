"""
Основной скрипт для агентно-центричного обучения модели Трансформер.
"""
import logging
import os
import time
import numpy as np

from config import Config, EvolutionConfig
from data_loader import load_text_from_directory
from model import Transformer
from nn_components.loss import SoftmaxCrossEntropy
from nn_components.lr_scheduler import cosine_decay_with_warmup
from optimizer import Adam, clip_gradients
from tokenizer import Tokenizer
from utils import load_checkpoint, save_checkpoint
from trainer import Trainer
from backend import set_backend

def setup_logging():
    """Настраивает логирование в файл и в консоль."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def load_and_prepare_data(config: EvolutionConfig, tokenizer_path: str):
    """Инициализирует токенизатор и загружает данные для обучения."""
    logging.info("Инициализация токенизатора и загрузка данных...")
    tokenizer = Tokenizer(tokenizer_path)

    all_text = load_text_from_directory(config.data_dir)
    if not all_text:
        raise ValueError(f"Не удалось загрузить текст из директории: {config.data_dir}")

    data_tokens = tokenizer.encode(all_text, add_special_tokens=True)
    split_idx = int(len(data_tokens) * (1 - config.validation_split))
    train_data, val_data = data_tokens[:split_idx], data_tokens[split_idx:]

    logging.info(f"Данные загружены. Словарь: {tokenizer.vocab_size}. "
                 f"Обучение: {len(train_data)} токенов. Валидация: {len(val_data)}.")
    return tokenizer, train_data, val_data

def initialize_components(config: Config, vocab_size: int):
    """Инициализирует модель, функцию потерь и оптимизатор."""
    logging.info("Инициализация модели, функции потерь и оптимизатора...")
    model = Transformer(vocab_size=vocab_size, model_config=config.model, ltm_config=config.ltm)
    policy_loss_fn = SoftmaxCrossEntropy()

    optim_config = config.optimizer.dict()
    max_norm = optim_config.pop('max_norm')
    optimizer = Adam(**optim_config)

    return model, policy_loss_fn, optimizer, max_norm

def main():
    """Orchestrates the agent-centric training process."""
    setup_logging()
    logging.info("--- Starting Agent-Centric Training ---")

    config = Config.from_json('config.json')
    set_backend(config.hardware.device)

    tokenizer, train_data, val_data = load_and_prepare_data(config.evolution, config.evolution.data_dir)
    model, loss_fn, optimizer, _ = initialize_components(config, tokenizer.vocab_size)

    trainer = Trainer(config, model, optimizer, loss_fn, tokenizer, train_data, val_data)

    start_epoch, current_step, best_val_loss, epochs_no_improve = 0, 0, float('inf'), 0
    if config.evolution.checkpoint_path and os.path.exists(config.evolution.checkpoint_path):
        state, _ = load_checkpoint(model, optimizer, config.evolution.checkpoint_path)
        if state:
            start_epoch, current_step, best_val_loss, epochs_no_improve = \
                state.get('epoch', 0), state.get('current_step', 0), \
                state.get('best_val_loss', float('inf')), state.get('epochs_no_improve', 0)
            logging.info(f"Resuming from epoch {start_epoch}, step {current_step}.")

    # --- Phase 1: Pre-training ---
    if start_epoch < config.evolution.pretrain_epochs:
        logging.info(f"--- Starting Pre-training Phase ({config.evolution.pretrain_epochs} epochs) ---")
        for epoch in range(start_epoch, config.evolution.pretrain_epochs):
            avg_loss, epoch_time, current_step = trainer.train_pretrain_epoch(current_step)
            val_loss = trainer.run_validation()
            logging.info(f"Epoch Pre-train {epoch+1}/{config.evolution.pretrain_epochs} | "
                         f"Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | "
                         f"LR: {optimizer.lr:.6f} | Time: {epoch_time:.2f}s")

            if val_loss < best_val_loss:
                best_val_loss, epochs_no_improve = val_loss, 0
                model.save_weights(config.evolution.best_model_path, config.dict())
            else:
                epochs_no_improve += 1

            if config.evolution.checkpoint_path:
                state = {'epoch': epoch + 1, 'current_step': current_step, 'best_val_loss': best_val_loss, 'epochs_no_improve': epochs_no_improve}
                save_checkpoint(model, optimizer, state, config.dict(), config.evolution.checkpoint_path)

            if epochs_no_improve >= config.evolution.early_stopping_patience:
                logging.warning("Early stopping during pre-training phase!")
                break
        start_epoch = config.evolution.pretrain_epochs

    # --- Phase 2: Evolutionary Cycle ---
    total_evolution_epochs = config.evolution.pretrain_epochs + config.evolution.evolution_epochs
    logging.info(f"--- Starting Evolution Phase ({config.evolution.evolution_epochs} cycles) ---")
    for epoch in range(start_epoch, total_evolution_epochs):
        epoch_time = trainer.run_evolution_cycle()
        val_loss = trainer.run_validation()
        logging.info(f"Epoch Evolution {epoch+1}/{total_evolution_epochs} | "
                     f"Base Model Val Loss: {val_loss:.4f} | Cycle Time: {epoch_time:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss, epochs_no_improve = val_loss, 0
            model.save_weights(config.evolution.best_model_path, config.dict())
            logging.info(f"New best base model saved with Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1

        if config.evolution.checkpoint_path:
            state = {'epoch': epoch + 1, 'current_step': current_step, 'best_val_loss': best_val_loss, 'epochs_no_improve': epochs_no_improve}
            save_checkpoint(model, optimizer, state, config.dict(), config.evolution.checkpoint_path)

        if epochs_no_improve >= config.evolution.early_stopping_patience:
            logging.info("Early stopping during evolution phase!")
            break

    logging.info("Training complete!")
    model.save_weights(config.evolution.weights_path, config.dict())

if __name__ == "__main__":
    main()
