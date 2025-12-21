"""
Основной скрипт для обучения модели Трансформер с использованием
контрастивного обучения для Value Head.
"""

import os
import time
import json
import logging
import numpy as np

from model import Transformer
from nn_components.loss import SoftmaxCrossEntropy, MarginRankingLoss
from optimizer import Adam, clip_gradients
from tokenizer import Tokenizer
from nn_components.lr_scheduler import cosine_decay_with_warmup
from utils import save_checkpoint, load_checkpoint

def get_batches(data, batch_size, seq_len):
    """Генератор батчей для обучения."""
    flat_data = np.array(data, dtype=np.int64)
    num_batches = len(flat_data) // (batch_size * seq_len)

    if num_batches == 0:
        raise ValueError("Недостаточно данных для создания хотя бы одного батча. "
                         "Попробуйте уменьшить batch_size или seq_len, или добавьте больше текста.")

    flat_data = flat_data[:num_batches * batch_size * seq_len]
    x = flat_data.reshape(batch_size, -1)
    y = np.roll(flat_data, -1).reshape(batch_size, -1)

    for i in range(0, x.shape[1], seq_len):
        x_batch = x[:, i:i + seq_len]
        y_batch = y[:, i:i + seq_len]
        yield x_batch, y_batch

def setup_logging():
    """Настраивает логирование в файл и в консоль."""
    # Remove existing handlers to avoid duplicate logs
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

def _get_candidate_pass_data(model, x, y, mask, policy_loss_fn):
    """Выполняет один проход (forward/backward) для кандидата."""
    model.train()
    model.zero_grad()

    logits, value = model.forward(x, mask)
    policy_loss = policy_loss_fn.forward(logits, y)
    dlogits = policy_loss_fn.backward()

    model.backward(dlogits, np.zeros_like(value))

    return {'value': value, 'loss': policy_loss, 'grads': model.get_gradients()}

def _accumulate_gradients(base_grads, new_grads):
    """Добавляет новые градиенты к существующим."""
    for key in base_grads:
        base_grads[key] += new_grads[key]
    return base_grads

def main():
    """Основной скрипт для обучения модели Трансформер."""
    setup_logging()
    logging.info("--- Запуск обучения модели Трансформер ---")

    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    model_config = config['model']
    train_config = config['training']
    optim_config = config['optimizer']
    scheduler_config = config.get('scheduler', {})

    logging.info("[Шаг 1/5] Инициализация токенизатора и загрузка данных...")
    tokenizer = Tokenizer(train_config['data_dir'])
    vocab_size = tokenizer.vocab_size

    # ... (data loading logic remains the same)
    all_text = ""
    for filename in os.listdir(train_config['data_dir']):
        if filename.endswith(".txt"):
            with open(os.path.join(train_config['data_dir'], filename), 'r', encoding='utf-8') as f:
                all_text += f.read()

    data_tokens = tokenizer.encode(all_text, add_special_tokens=True)
    val_split = train_config.get('validation_split', 0.0)
    split_idx = int(len(data_tokens) * (1 - val_split))
    train_data, val_data = data_tokens[:split_idx], data_tokens[split_idx:]

    logging.info("Данные успешно загружены. Размер словаря: %d. Обучающая выборка: %d токенов. Валидационная: %d.",
                 vocab_size, len(train_data), len(val_data))

    logging.info("[Шаг 2/5] Инициализация модели, функций потерь и оптимизатора...")
    model = Transformer(vocab_size=vocab_size, **model_config)
    policy_loss_fn = SoftmaxCrossEntropy()
    value_loss_fn = MarginRankingLoss(margin=train_config.get('contrastive_margin', 1.0))
    max_norm = optim_config.pop('max_norm')
    optimizer = Adam(model.get_named_params(), **optim_config)

    logging.info("[Шаг 3/5] Настройка цикла обучения...")
    start_epoch, current_step = 0, 0
    checkpoint_path = train_config.get('checkpoint_path')
    if checkpoint_path and os.path.exists(checkpoint_path):
        logging.info("Загрузка с контрольной точки: %s", checkpoint_path)
        state, _ = load_checkpoint(model, optimizer, checkpoint_path)
        if state:
            start_epoch, current_step = state['epoch'] + 1, state['current_step']
            logging.info("Обучение возобновлено с эпохи %d, шаг %d.", start_epoch, current_step)

    grad_accum_steps = train_config.get('gradient_accumulation_steps', 1)
    num_batches = len(train_data) // (train_config['batch_size'] * train_config['seq_len'])
    training_steps = (num_batches // grad_accum_steps) * train_config['epochs']

    logging.info("Всего шагов оптимизации: %d. Накопление градиентов: %d шаг(а).", training_steps, grad_accum_steps)

    logging.info("[Шаг 4/5] Начало цикла обучения...")
    for epoch in range(start_epoch, train_config['epochs']):
        start_time = time.time()
        total_loss, total_policy_loss, total_value_loss = 0, 0, 0

        batch_iterator = get_batches(train_data, train_config['batch_size'], train_config['seq_len'])
        grad_accumulator = {key: np.zeros_like(param) for key, (param, _) in model.get_named_params()['embedding'].get_trainable_params().items()}

        for i, (x, y) in enumerate(batch_iterator):
            mask = np.triu(np.ones((x.shape[1], x.shape[1])), k=1).astype(bool)

            # 1. Contrastive Step
            candidates = [_get_candidate_pass_data(model, x, y, mask, policy_loss_fn) for _ in range(train_config.get('num_candidates', 4))]
            candidates.sort(key=lambda c: c['loss'])
            best, worst = candidates[0], candidates[-1]

            # 2. Calculate losses
            policy_loss = best['loss']
            value_loss = value_loss_fn.forward(best['value'], worst['value'])
            d_good_v, d_bad_v = value_loss_fn.backward()

            # 3. Accumulate gradients
            final_grads = best['grads']

            model.zero_grad(); model.eval(); _, _ = model.forward(x, mask)
            model.backward(np.zeros_like(final_grads['embedding.W']), d_good_v)
            final_grads = _accumulate_gradients(final_grads, model.get_gradients())

            model.zero_grad(); model.eval(); _, _ = model.forward(x, mask)
            model.backward(np.zeros_like(final_grads['embedding.W']), d_bad_v)
            final_grads = _accumulate_gradients(final_grads, model.get_gradients())

            grad_accumulator = _accumulate_gradients(grad_accumulator, final_grads)

            total_loss += (policy_loss + value_loss)
            total_policy_loss += policy_loss
            total_value_loss += value_loss

            if (i + 1) % grad_accum_steps == 0:
                # Normalize gradients
                for key in grad_accumulator:
                    grad_accumulator[key] /= grad_accum_steps

                model.set_gradients(grad_accumulator)
                clip_gradients(model.get_named_params(), max_norm)

                new_lr = cosine_decay_with_warmup(current_step, training_steps, **scheduler_config)
                optimizer.lr = new_lr
                optimizer.step()

                model.zero_grad()
                grad_accumulator = {key: np.zeros_like(val) for key, val in grad_accumulator.items()}
                current_step += 1

        # --- End of Epoch ---
        avg_loss = total_loss / num_batches
        avg_policy = total_policy_loss / num_batches
        avg_value = total_value_loss / num_batches
        epoch_time = time.time() - start_time

        log_msg = (f"Эпоха {epoch+1}/{train_config['epochs']} | Потери: {avg_loss:.4f} "
                   f"(Policy: {avg_policy:.4f}, Value: {avg_value:.4f}) | "
                   f"LR: {optimizer.lr:.6f} | Время: {epoch_time:.2f}с")

        if len(val_data) > 0:
            val_loss = run_validation(model, val_data, policy_loss_fn, train_config)
            log_msg += f" | Val Потери: {val_loss:.4f}"

        logging.info(log_msg)

        if checkpoint_path:
            save_checkpoint(model, optimizer, epoch, current_step, config, checkpoint_path)

    logging.info("[Шаг 5/5] Обучение завершено!")
    model.save_weights(train_config['weights_path'], config)

def run_validation(model, val_data, policy_loss_fn, config):
    """Выполняет проход по валидационным данным и возвращает средние потери."""
    model.eval()
    total_val_loss, val_batches = 0, 0

    batch_iterator = get_batches(val_data, config['batch_size'], config['seq_len'])

    for x, y in batch_iterator:
        mask = np.triu(np.ones((x.shape[1], x.shape[1])), k=1).astype(bool)
        logits, _ = model.forward(x, mask)
        total_val_loss += policy_loss_fn.forward(logits, y)
        val_batches += 1

    model.train()
    return total_val_loss / val_batches if val_batches > 0 else 0.0

if __name__ == "__main__":
    main()
