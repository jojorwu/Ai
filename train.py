import numpy as np
import os
import time
import json
import logging
from model import Transformer
from nn_components.loss import SoftmaxCrossEntropy
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

    # Форматтер
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Обработчик для файла
    file_handler = logging.FileHandler('training.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Обработчик для консоли
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

    # --- 1. Загрузка конфигурации ---
    with open('config.json', 'r') as f:
        config = json.load(f)

    model_config = config['model']
    train_config = config['training']
    optim_config = config['optimizer']
    scheduler_config = config.get('scheduler', {})

    # --- 2. Подготовка данных ---
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

    data_tokens = tokenizer.encode(all_text)
    logging.info(f"Данные успешно загружены. Размер словаря: {vocab_size}, Всего токенов: {len(data_tokens)}")

    # --- 3. Инициализация модели и оптимизатора ---
    logging.info("[Шаг 2/4] Инициализация модели, функции потерь и оптимизатора...")
    model = Transformer(vocab_size=vocab_size, **model_config)
    loss_fn = SoftmaxCrossEntropy()

    # Отделяем max_norm от параметров Adam
    max_norm = optim_config.pop('max_norm')
    optimizer = Adam(model.get_named_params(), **optim_config)

    mask = np.triu(np.ones((train_config['seq_len'], train_config['seq_len'])), k=1).astype(bool)

    model.train()

    start_epoch = 0
    current_step = 0

    # --- 4. Загрузка контрольной точки (если есть) ---
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

    # --- 5. Настройка планировщика и цикл обучения ---
    gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', 1)

    num_batches = len(data_tokens) // (train_config['batch_size'] * train_config['seq_len'])
    training_steps = (num_batches // gradient_accumulation_steps) * train_config['epochs']

    max_lr = optim_config['learning_rate']
    min_lr = scheduler_config.get('min_lr', 1e-6)
    warmup_steps = scheduler_config.get('warmup_steps', 0)

    logging.info(f"Всего шагов оптимизации: {training_steps}")
    logging.info(f"Накопление градиентов: {gradient_accumulation_steps} шаг(а)")

    if start_epoch == 0:
        model.zero_grad()

    for epoch in range(start_epoch, train_config['epochs']):
        start_time = time.time()
        total_loss = 0

        # Обертка для enumerate, чтобы получить индекс батча
        batch_iterator = get_batches(data_tokens, train_config['batch_size'], train_config['seq_len'])

        for i, (x, y) in enumerate(batch_iterator):
            # 1. Прямой и обратный проход
            logits = model.forward(x, mask)
            loss = loss_fn.forward(logits, y)

            # 2. Нормализация потерь для усреднения градиентов
            loss = loss / gradient_accumulation_steps
            total_loss += loss.item() # Накапливаем фактическую потерю

            dlogits = loss_fn.backward()
            model.backward(dlogits) # Градиенты накапливаются внутри модели

            # 3. Обновление весов после накопления
            if (i + 1) % gradient_accumulation_steps == 0:
                # Обновление learning rate происходит перед шагом оптимизатора
                new_lr = cosine_decay_with_warmup(current_step, training_steps, warmup_steps, max_lr, min_lr)
                optimizer.lr = new_lr

                clip_gradients(model.get_named_params(), max_norm)
                optimizer.step()
                model.zero_grad() # Очистка градиентов для следующего цикла накопления

                current_step += 1

        # Расчет и вывод средней потери за эпоху
        # Делим на количество шагов оптимизации, а не на количество батчей
        epoch_loss = total_loss / (num_batches / gradient_accumulation_steps)
        epoch_time = time.time() - start_time
        logging.info(f"Эпоха {epoch+1}/{train_config['epochs']} | Потери: {epoch_loss:.4f} | LR: {optimizer.lr:.6f} | Время: {epoch_time:.2f}с")

        # Сохранение контрольной точки
        if checkpoint_path:
            save_checkpoint(model, optimizer, epoch, current_step, config, checkpoint_path)
            logging.info(f"Контрольная точка сохранена в {checkpoint_path}")

    logging.info("[Шаг 6/6] Обучение завершено!")

    model.save_weights(train_config['weights_path'], config)


if __name__ == "__main__":
    main()
