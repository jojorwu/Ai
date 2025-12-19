import numpy as np
import os
import time
from model import Transformer
from nn_components.loss import SoftmaxCrossEntropy
from optimizer import SGD
from tokenizer import Tokenizer

def get_batches(data, batch_size, seq_len):
    """Генератор батчей для обучения."""
    # Превращаем данные в один длинный массив
    flat_data = np.array(data, dtype=np.int64)
    num_batches = len(flat_data) // (batch_size * seq_len)

    if num_batches == 0:
        raise ValueError("Недостаточно данных для создания хотя бы одного батча. "
                         "Попробуйте уменьшить batch_size или seq_len, или добавьте больше текста.")

    # Обрезаем данные, чтобы они были кратны размеру батча
    flat_data = flat_data[:num_batches * batch_size * seq_len]

    # Формируем x и y
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

    # --- 1. Гиперпараметры ---
    # Модель
    vocab_size = None # Определится токенизатором
    d_model = 64
    num_layers = 4
    num_heads = 4
    d_ff = 256
    max_seq_len = 128

    # Обучение
    epochs = 10
    batch_size = 2  # Уменьшено для небольшого набора данных
    seq_len = 32  # Уменьшено для небольшого набора данных
    learning_rate = 1e-3
    data_dir = "data"

    # --- 2. Подготовка данных ---
    print("\n[Шаг 1/4] Инициализация токенизатора и загрузка данных...")
    if not os.path.exists(data_dir) or not any(f.endswith('.txt') for f in os.listdir(data_dir)):
        print(f"Ошибка: Директория '{data_dir}' не найдена или не содержит .txt файлов.")
        return

    tokenizer = Tokenizer(data_dir)
    vocab_size = tokenizer.vocab_size

    all_text = ""
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                all_text += f.read()

    data_tokens = tokenizer.encode(all_text)
    print(f"Данные успешно загружены. Размер словаря: {vocab_size}, Всего токенов: {len(data_tokens)}")

    # --- 3. Инициализация модели и оптимизатора ---
    print("\n[Шаг 2/4] Инициализация модели, функции потерь и оптимизатора...")
    model = Transformer(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len)
    loss_fn = SoftmaxCrossEntropy()
    optimizer = SGD(model.get_params(), learning_rate)

    # Создаем Causal маску
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)

    # --- 4. Цикл обучения ---
    print("\n[Шаг 3/4] Начало цикла обучения...")
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        batch_count = 0

        for x, y in get_batches(data_tokens, batch_size, seq_len):
            # Forward pass
            logits = model.forward(x, mask)
            loss = loss_fn.forward(logits, y)

            # Backward pass
            dlogits = loss_fn.backward()
            model.backward(dlogits)

            # Update weights
            optimizer.step()

            total_loss += loss
            batch_count += 1

        epoch_loss = total_loss / batch_count
        epoch_time = time.time() - start_time
        print(f"Эпоха {epoch+1}/{epochs} | Потери: {epoch_loss:.4f} | Время: {epoch_time:.2f}с")

    print("\n[Шаг 4/4] Обучение завершено!")

    # --- 5. Сохранение весов ---
    weights_path = "model_weights.npz"
    model.save_weights(weights_path)


if __name__ == "__main__":
    main()
