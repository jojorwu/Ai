import numpy as np
import os
from model import Transformer
from tokenizer import Tokenizer

def main():
    """
    Скрипт для генерации текста с использованием обученной модели.
    """
    print("--- Запуск генерации текста ---")

    # --- 1. Параметры ---
    # Эти параметры должны совпадать с теми, на которых обучалась модель
    d_model = 64
    num_layers = 4
    num_heads = 4
    d_ff = 256
    max_seq_len = 128
    data_dir = "data"
    weights_path = "model_weights.npz"

    # Параметры генерации
    start_text = "Привет"
    max_len = 50 # Количество символов для генерации
    temperature = 0.8 # Чем выше, тем более случайный текст

    # --- 2. Загрузка ---
    print("\n[Шаг 1/3] Загрузка токенизатора и модели...")
    if not os.path.exists(weights_path):
        print(f"Ошибка: Файл с весами '{weights_path}' не найден. "
              "Пожалуйста, сначала запустите train.py для обучения модели.")
        return

    tokenizer = Tokenizer(data_dir)
    vocab_size = tokenizer.vocab_size

    model = Transformer(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len)
    model.load_weights(weights_path)
    print("Модель и веса успешно загружены.")

    # --- 3. Генерация ---
    print(f"\n[Шаг 2/3] Генерация текста, начиная с фразы: '{start_text}'...")

    # Кодируем начальный текст
    start_tokens = tokenizer.encode(start_text)

    # Генерируем новые токены
    generated_tokens = model.generate(start_tokens, max_len=max_len, temperature=temperature)

    # Декодируем результат
    generated_text = tokenizer.decode(generated_tokens.tolist())

    print("\n[Шаг 3/3] Результат:")
    print("="*20)
    print(start_text + generated_text)
    print("="*20)

if __name__ == "__main__":
    main()
