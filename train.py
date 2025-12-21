import numpy as np
import os
import time
import json
import logging
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
        x_batch = x[:, i:i+seq_len]
        y_batch = y[:, i:i+seq_len]
        yield x_batch, y_batch

def setup_logging():
    """Настраивает логирование в файл и в консоль."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler('training.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def main():
    """
    Основной скрипт для обучения модели Трансформер.
    """
    setup_logging()
    logging.info("--- Запуск обучения модели Трансформер ---")

    with open('config.json', 'r') as f:
        config = json.load(f)

    model_config = config['model']
    train_config = config['training']
    optim_config = config['optimizer']
    scheduler_config = config.get('scheduler', {})

    logging.info("[Шаг 1/4] Инициализация токенизатора и загрузка данных...")
    if not os.path.exists(train_config['data_dir']) or not any(f.endswith('.txt') for f in os.listdir(train_config['data_dir'])):
        logging.error(f"Ошибка: Директория '{train_config['data_dir']}' не найдена или не содержит .txt файлов.")
        return

    tokenizer = Tokenizer(train_config['data_dir'])
    vocab_size = tokenizer.vocab_size

    all_text = ""
    for filename in os.listdir(train_config['data_dir']):
        if filename.endswith(".txt"):
            with open(os.path.join(train_config['data_dir'], filename), 'r', encoding='utf-8') as f:
                all_text += f.read()

    data_tokens = tokenizer.encode(all_text, add_special_tokens=True)

    val_split = train_config.get('validation_split', 0.0)
    split_idx = int(len(data_tokens) * (1 - val_split))
    train_data = data_tokens[:split_idx]
    val_data = data_tokens[split_idx:]

    logging.info(f"Данные успешно загружены. Размер словаря: {vocab_size}, всего токенов: {len(data_tokens)}")
    logging.info(f"Обучающая выборка: {len(train_data)} токенов, Валидационная выборка: {len(val_data)} токенов")

    logging.info("[Шаг 2/4] Инициализация модели, функции потерь и оптимизатора...")
    model = Transformer(vocab_size=vocab_size, **model_config)
    policy_loss_fn = SoftmaxCrossEntropy()
    value_loss_fn = MarginRankingLoss(margin=train_config.get('contrastive_margin', 1.0))

    max_norm = optim_config.pop('max_norm')
    optimizer = Adam(model.get_named_params(), **optim_config)

    mask = np.triu(np.ones((train_config['seq_len'], train_config['seq_len'])), k=1).astype(bool)

    start_epoch = 0
    current_step = 0

    checkpoint_path = train_config.get('checkpoint_path')
    if checkpoint_path and os.path.exists(checkpoint_path):
        logging.info(f"[Шаг 3/4] Обнаружена контрольная точка. Загрузка...")
        training_state, _ = load_checkpoint(model, optimizer, checkpoint_path)
        if training_state:
            start_epoch = training_state['epoch'] + 1
            current_step = training_state['current_step']
            logging.info(f"Обучение возобновлено с эпохи {start_epoch}, шаг {current_step}.")
    else:
        logging.info("[Шаг 3/4] Начало цикла обучения...")

    gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', 1)
    num_batches = len(train_data) // (train_config['batch_size'] * train_config['seq_len'])
    training_steps = (num_batches // gradient_accumulation_steps) * train_config['epochs']
    max_lr = optim_config['learning_rate']
    min_lr = scheduler_config.get('min_lr', 1e-6)
    warmup_steps = scheduler_config.get('warmup_steps', 0)

    logging.info(f"Всего шагов оптимизации: {training_steps}")
    logging.info(f"Накопление градиентов: {gradient_accumulation_steps} шаг(а)")

    num_candidates = train_config.get('num_candidates', 4)

    for epoch in range(start_epoch, train_config['epochs']):
        start_time = time.time()
        total_loss = 0
        batch_iterator = get_batches(train_data, train_config['batch_size'], train_config['seq_len'])

        for i, (x, y) in enumerate(batch_iterator):
            model.zero_grad()

            # --- Contrastive Value and Policy Training ---
            candidate_data = []

            for _ in range(num_candidates):
                model.train() # Ensure dropout is on

                # 1. Forward pass
                logits, value = model.forward(x, mask)

                # 2. Calculate policy loss
                policy_loss = policy_loss_fn.forward(logits, y)

                # 3. Backward pass for policy loss
                dlogits = policy_loss_fn.backward()
                model.backward(dlogits, np.zeros_like(value))

                # 4. Store gradients and other data
                candidate_data.append({
                    'value': value,
                    'loss': policy_loss,
                    'grads': model.get_gradients()
                })
                model.zero_grad()

            # Identify best and worst candidates
            candidate_data.sort(key=lambda c: c['loss'])
            best_candidate = candidate_data[0]
            worst_candidate = candidate_data[-1]

            # Get value predictions for best and worst
            good_value = best_candidate['value']
            bad_value = worst_candidate['value']

            # Calculate contrastive loss and gradients
            value_loss = value_loss_fn.forward(good_value, bad_value)
            d_good_value, d_bad_value = value_loss_fn.backward()

            # Get the stored gradients for the best and worst candidates
            best_grads = best_candidate['grads']
            worst_grads = worst_candidate['grads']

            # --- Accumulate Gradients ---
            # Set the model's gradients to the 'best' candidate's policy gradients
            model.set_gradients(best_grads)
            # Re-run forward pass deterministically to set state for value backward pass
            model.eval()
            _, _ = model.forward(x, mask)
            # Accumulate the gradient from the 'good' value
            model.backward(np.zeros_like(logits), d_good_value)

            # Get the combined gradients for the best candidate
            accumulated_grads = model.get_gradients()

            # Set the model's gradients to the 'worst' candidate's policy gradients
            model.set_gradients(worst_grads)
            # Re-run forward pass deterministically
            model.eval()
            _, _ = model.forward(x, mask)
            # Accumulate the gradient from the 'bad' value
            model.backward(np.zeros_like(logits), d_bad_value)

            # Add the worst candidate's gradients to the accumulated gradients
            worst_combined_grads = model.get_gradients()
            for key in accumulated_grads:
                accumulated_grads[key] += worst_combined_grads[key]

            # Set the final accumulated gradients to the model
            model.set_gradients(accumulated_grads)

            loss = best_candidate['loss'] + value_loss
            loss = loss / gradient_accumulation_steps
            total_loss += loss.item()

            if (i + 1) % gradient_accumulation_steps == 0:
                new_lr = cosine_decay_with_warmup(current_step, training_steps, warmup_steps, max_lr, min_lr)
                optimizer.lr = new_lr
                clip_gradients(model.get_named_params(), max_norm)
                optimizer.step()
                model.zero_grad()
                current_step += 1

        epoch_loss = total_loss / (num_batches / gradient_accumulation_steps)
        epoch_time = time.time() - start_time

        if len(val_data) > 0:
            val_loss = run_validation(model, val_data, policy_loss_fn, train_config)
            logging.info(f"Эпоха {epoch+1}/{train_config['epochs']} | Потери: {epoch_loss:.4f} | Val Потери: {val_loss:.4f} | LR: {optimizer.lr:.6f} | Время: {epoch_time:.2f}с")
        else:
            logging.info(f"Эпоха {epoch+1}/{train_config['epochs']} | Потери: {epoch_loss:.4f} | LR: {optimizer.lr:.6f} | Время: {epoch_time:.2f}с")

        if checkpoint_path:
            save_checkpoint(model, optimizer, epoch, current_step, config, checkpoint_path)

    logging.info("[Шаг 6/6] Обучение завершено!")
    model.save_weights(train_config['weights_path'], config)

def run_validation(model, val_data, policy_loss_fn, config):
    """Выполняет проход по валидационным данным и возвращает средние потери."""
    model.eval()
    total_val_loss = 0
    val_batches = 0

    batch_iterator = get_batches(val_data, config['batch_size'], config['seq_len'])
    mask = np.triu(np.ones((config['seq_len'], config['seq_len'])), k=1).astype(bool)

    for x, y in batch_iterator:
        logits, _ = model.forward(x, mask)
        loss = policy_loss_fn.forward(logits, y)
        total_val_loss += loss
        val_batches += 1

    model.train()

    if val_batches == 0:
        return 0.0

    return total_val_loss / val_batches

if __name__ == "__main__":
    main()
