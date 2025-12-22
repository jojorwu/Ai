"""
Основной скрипт для обучения модели Трансформер.
"""
import logging
import os
import time
import numpy as np

from config import Config, TrainingConfig
from data_loader import load_text_from_directory
from model import Transformer
from nn_components.loss import MarginRankingLoss, SoftmaxCrossEntropy
from nn_components.lr_scheduler import cosine_decay_with_warmup
from optimizer import Adam, clip_gradients
from tokenizer import Tokenizer
from utils import load_checkpoint, save_checkpoint

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

def load_and_prepare_data(config: TrainingConfig):
    """Инициализирует токенизатор и загружает данные для обучения."""
    logging.info("Инициализация токенизатора и загрузка данных...")
    tokenizer = Tokenizer(config.data_dir)
    all_text = load_text_from_directory(config.data_dir)
    if not all_text:
        raise ValueError("Не удалось загрузить текст из директории.")
    data_tokens = tokenizer.encode(all_text, add_special_tokens=True)
    split_idx = int(len(data_tokens) * (1 - config.validation_split))
    train_data, val_data = data_tokens[:split_idx], data_tokens[split_idx:]
    logging.info(f"Данные загружены. Словарь: {tokenizer.vocab_size}. "
                 f"Обучение: {len(train_data)} токенов. Валидация: {len(val_data)}.")
    return tokenizer, train_data, val_data

def initialize_components(config: Config, vocab_size: int):
    """Инициализирует модель, функции потерь и оптимизатор."""
    logging.info("Инициализация модели, функций потерь и оптимизатора...")
    model = Transformer(vocab_size=vocab_size, **config.model.dict())
    policy_loss_fn = SoftmaxCrossEntropy()
    value_loss_fn = MarginRankingLoss(margin=config.training.contrastive_margin)
    optim_config = config.optimizer.dict()
    max_norm = optim_config.pop('max_norm')
    optimizer = Adam(model.get_named_params(), **optim_config)
    return model, policy_loss_fn, value_loss_fn, optimizer, max_norm

def get_batches(data, batch_size, seq_len):
    """Генератор батчей для обучения."""
    flat_data = np.array(data, dtype=np.int64)
    num_batches = len(flat_data) // (batch_size * seq_len)
    if num_batches == 0:
        raise ValueError("Недостаточно данных для создания хотя бы одного батча.")
    flat_data = flat_data[:num_batches * batch_size * seq_len]
    x = flat_data.reshape(batch_size, -1)
    y = np.roll(flat_data, -1).reshape(batch_size, -1)
    for i in range(0, x.shape[1], seq_len):
        yield x[:, i:i + seq_len], y[:, i:i + seq_len]

def run_validation(model: Transformer, val_data: list, loss_fn: SoftmaxCrossEntropy, config: TrainingConfig):
    """Запускает валидацию модели на отдельном наборе данных."""
    model.eval()
    total_loss, num_batches = 0, 0
    batch_iterator = get_batches(val_data, config.batch_size, config.seq_len)
    for x, y in batch_iterator:
        mask = np.triu(np.ones((x.shape[1], x.shape[1])), k=1).astype(bool)
        logits, _ = model.forward(x, mask)
        total_loss += loss_fn.forward(logits, y)
        num_batches += 1
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0

def train_epoch_stage1(model: Transformer, data: list, policy_loss_fn, optimizer, configs, max_norm, current_step):
    """Выполняет одну эпоху обучения для этапа 1 (только policy pre-training)."""
    train_config, scheduler_config = configs
    start_time = time.time()
    total_policy_loss = 0

    batch_iterator = get_batches(data, train_config.batch_size, train_config.seq_len)
    num_batches = len(data) // (train_config.batch_size * train_config.seq_len)
    training_steps = (num_batches // train_config.gradient_accumulation_steps) * train_config.epochs

    optimizer.zero_grad()
    for i, (x, y) in enumerate(batch_iterator):
        model.train()
        mask = np.triu(np.ones((x.shape[1], x.shape[1])), k=1).astype(bool)

        # Прямой проход
        logits, _ = model.forward(x, mask)
        policy_loss = policy_loss_fn.forward(logits, y)
        total_policy_loss += policy_loss

        # Обратный проход
        dlogits = policy_loss_fn.backward()
        # Для dvalues передаем нули, так как value head не используется на этом этапе
        dvalues = np.zeros((x.shape[0], 1))
        model.backward(dlogits, dvalues)

        # Обновление весов с накоплением градиентов
        if (i + 1) % train_config.gradient_accumulation_steps == 0:
            clip_gradients(model.get_named_params(flat=False), max_norm)
            new_lr = cosine_decay_with_warmup(current_step, training_steps, **scheduler_config.dict())
            optimizer.lr = new_lr
            optimizer.step()
            optimizer.zero_grad()
            current_step += 1

    avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0
    epoch_time = time.time() - start_time
    return avg_policy_loss, avg_policy_loss, 0.0, epoch_time, current_step


def train_epoch_stage2(model: Transformer, data: list, loss_fns, optimizer, configs, max_norm, current_step):
    """Выполняет одну эпоху обучения для этапа 2 (contrastive fine-tuning)."""
    train_config, scheduler_config = configs
    policy_loss_fn, value_loss_fn = loss_fns
    start_time = time.time()
    total_policy_loss, total_value_loss = 0, 0

    batch_iterator = get_batches(data, train_config.batch_size, train_config.seq_len)
    num_batches = len(data) // (train_config.batch_size * train_config.seq_len)
    training_steps = (num_batches // train_config.gradient_accumulation_steps) * train_config.epochs

    optimizer.zero_grad()
    for i, (x, y) in enumerate(batch_iterator):
        model.train()

        # Расширение батча для векторизации
        num_candidates = train_config.num_candidates
        original_batch_size = x.shape[0]
        x_expanded = np.repeat(x, num_candidates, axis=0)
        y_expanded = np.repeat(y, num_candidates, axis=0)
        mask = np.triu(np.ones((x.shape[1], x.shape[1])), k=1).astype(bool)

        # Прямой проход для всех кандидатов
        logits, values = model.forward(x_expanded, mask)

        # Векторизованный выбор лучших/худших
        candidate_losses = policy_loss_fn.forward(logits, y_expanded, reduction='none')
        candidate_losses_reshaped = candidate_losses.reshape(original_batch_size, num_candidates)
        best_indices = np.argmin(candidate_losses_reshaped, axis=1)
        worst_indices = np.argmax(candidate_losses_reshaped, axis=1)
        base_indices = np.arange(original_batch_size) * num_candidates
        best_global_indices = base_indices + best_indices
        worst_global_indices = base_indices + worst_indices

        # Расчет потерь
        policy_loss = np.mean(candidate_losses[best_global_indices])
        value_loss = value_loss_fn.forward(values[best_global_indices], values[worst_global_indices])
        total_policy_loss += policy_loss
        total_value_loss += value_loss

        # Расчет и маскирование градиентов
        dlogits = policy_loss_fn.backward()
        policy_mask = np.zeros(x_expanded.shape[0], dtype=bool)
        policy_mask[best_global_indices] = True
        dlogits[~policy_mask] = 0

        d_good_v, d_bad_v = value_loss_fn.backward()
        dvalues = np.zeros_like(values)
        dvalues[best_global_indices] = d_good_v
        dvalues[worst_global_indices] += d_bad_v

        # Обратный проход
        model.backward(dlogits, dvalues)

        # Обновление весов
        if (i + 1) % train_config.gradient_accumulation_steps == 0:
            clip_gradients(model.get_named_params(flat=False), max_norm)
            new_lr = cosine_decay_with_warmup(current_step, training_steps, **scheduler_config.dict())
            optimizer.lr = new_lr
            optimizer.step()
            optimizer.zero_grad()
            current_step += 1

    avg_policy = total_policy_loss / num_batches if num_batches > 0 else 0
    avg_value = total_value_loss / num_batches if num_batches > 0 else 0
    avg_loss = avg_policy + avg_value
    epoch_time = time.time() - start_time
    return avg_loss, avg_policy, avg_value, epoch_time, current_step


def main():
    """Основной скрипт для обучения модели."""
    setup_logging()
    logging.info("--- Запуск обучения модели Трансформер ---")
    config = Config.from_json('config.json')
    tokenizer, train_data, val_data = load_and_prepare_data(config.training)
    model, policy_loss_fn, value_loss_fn, optimizer, max_norm = initialize_components(config, tokenizer.vocab_size)

    start_epoch, current_step = 0, 0
    if config.training.checkpoint_path and os.path.exists(config.training.checkpoint_path):
        state, _ = load_checkpoint(model, optimizer, config.training.checkpoint_path)
        if state:
            start_epoch, current_step = state['epoch'], state['current_step']
            logging.info(f"Возобновление с эпохи {start_epoch}, шаг {current_step}.")

    logging.info(f"Начало цикла обучения. Этап: {config.training.training_stage}")
    for epoch in range(start_epoch, config.training.epochs):
        if config.training.training_stage == 1:
            avg_loss, avg_policy, avg_value, epoch_time, current_step = train_epoch_stage1(
                model, train_data, policy_loss_fn, optimizer,
                (config.training, config.scheduler), max_norm, current_step
            )
        elif config.training.training_stage == 2:
            avg_loss, avg_policy, avg_value, epoch_time, current_step = train_epoch_stage2(
                model, train_data, (policy_loss_fn, value_loss_fn), optimizer,
                (config.training, config.scheduler), max_norm, current_step
            )
        else:
            raise ValueError(f"Неизвестный этап обучения: {config.training.training_stage}")

        log_msg_parts = [
            f"Эпоха {epoch+1}/{config.training.epochs}",
            f"Потери: {avg_loss:.4f}",
            f"(Policy: {avg_policy:.4f}"
        ]
        if config.training.training_stage == 2:
            log_msg_parts.append(f", Value: {avg_value:.4f})")
        else:
            log_msg_parts.append(")")
        log_msg_parts.extend([
            f"LR: {optimizer.lr:.6f}",
            f"Время: {epoch_time:.2f}с"
        ])

        if len(val_data) > 0:
            val_loss = run_validation(model, val_data, policy_loss_fn, config.training)
            log_msg_parts.append(f"Val Потери: {val_loss:.4f}")

        logging.info(" | ".join(log_msg_parts))

        if config.training.checkpoint_path:
            save_checkpoint(model, optimizer, epoch + 1, current_step, config.dict(), config.training.checkpoint_path)

    logging.info("Обучение завершено!")
    model.save_weights(config.training.weights_path, config.dict())

if __name__ == "__main__":
    main()
