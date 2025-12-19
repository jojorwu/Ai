import numpy as np
import os
import time
import json
from model import Transformer
from nn_components.loss import SoftmaxCrossEntropy
from optimizer import Adam, clip_gradients
from tokenizer import Tokenizer

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

def main():
    """
    Основной скрипт для обучения модели Трансформер.
    """
    print("--- Запуск обучения модели Трансформер ---")

    # --- 1. Загрузка конфигурации ---
    with open('config.json', 'r') as f:
        config = json.load(f)

    model_config = config['model']
    train_config = config['training']
    optim_config = config['optimizer']

    # --- 2. Подготовка данных ---
    print("\n[Шаг 1/4] Инициализация токенизатора и загрузка данных...")
    if not os.path.exists(train_config['data_dir']) or not any(f.endswith('.txt') for f in os.listdir(train_config['data_dir'])):
        print(f"Ошибка: Директория '{train_config['data_dir']}' не найдена или не содержит .txt файлов.")
        return

    tokenizer = Tokenizer(train_config['data_dir'])
    vocab_size = tokenizer.vocab_size

    all_text = ""
    for filename in os.listdir(train_config['data_dir']):
        if filename.endswith(".txt"):
            with open(os.path.join(train_config['data_dir'], filename), 'r', encoding='utf-8') as f:
                all_text += f.read()

    data_tokens = tokenizer.encode(all_text)
    print(f"Данные успешно загружены. Размер словаря: {vocab_size}, Всего токенов: {len(data_tokens)}")

    # --- 3. Инициализация модели и оптимизатора ---
    print("\n[Шаг 2/4] Инициализация модели, функции потерь и оптимизатора...")
    model = Transformer(vocab_size=vocab_size, **model_config)
    loss_fn = SoftmaxCrossEntropy()

    # Отделяем max_norm от параметров Adam
    max_norm = optim_config.pop('max_norm')
    optimizer = Adam(model.get_named_params(), **optim_config)

    mask = np.triu(np.ones((train_config['seq_len'], train_config['seq_len'])), k=1).astype(bool)

    model.train()

    # --- 4. Цикл обучения ---
    print("\n[Шаг 3/4] Начало цикла обучения...")
    for epoch in range(train_config['epochs']):
        start_time = time.time()
        total_loss = 0
        batch_count = 0

        for x, y in get_batches(data_tokens, train_config['batch_size'], train_config['seq_len']):
            logits = model.forward(x, mask)
            loss = loss_fn.forward(logits, y)

            dlogits = loss_fn.backward()
            model.backward(dlogits)

            # Обрезка градиентов
            clip_gradients(model.get_named_params(), max_norm)

            optimizer.step()

            total_loss += loss
            batch_count += 1

        epoch_loss = total_loss / batch_count
        epoch_time = time.time() - start_time
        print(f"Эпоха {epoch+1}/{train_config['epochs']} | Потери: {epoch_loss:.4f} | Время: {epoch_time:.2f}с")

    print("\n[Шаг 4/4] Обучение завершено!")

    model.save_weights(train_config['weights_path'], config)

if __name__ == "__main__":
    main()
