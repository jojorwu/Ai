import numpy as np
import os
import json
from model import Transformer
from tokenizer import Tokenizer

def main():
    """
    Скрипт для генерации текста с использованием обученной модели.
    """
    print("--- Запуск генерации текста ---")

    # --- 1. Загрузка конфигурации ---
    with open('config.json', 'r') as f:
        config = json.load(f)

    train_config = config['training']
    gen_config = config['generation']
    weights_path = train_config['weights_path']
    data_dir = train_config['data_dir']

    # --- 2. Загрузка ---
    print("\n[Шаг 1/3] Загрузка токенизатора и модели...")
    if not os.path.exists(weights_path):
        print(f"Ошибка: Файл с весами '{weights_path}' не найден. "
              "Пожалуйста, сначала запустите train.py для обучения модели.")
        return

    tokenizer = Tokenizer(data_dir)
    vocab_size = tokenizer.vocab_size

    # Загружаем модель (архитектуру и веса) из файла
    model, _ = Transformer.load_model(weights_path, vocab_size)
    print("Модель и веса успешно загружены.")

    # Переключаем модель в режим генерации
    model.eval()

    # --- 3. Генерация ---
    start_text = gen_config['start_text']
    print(f"\n[Шаг 2/3] Генерация текста, начиная с фразы: '{start_text}'...")

    start_tokens = tokenizer.encode(start_text)

    generated_tokens = model.generate(
        start_tokens,
        max_len=gen_config.get('max_len', 50),
        temperature=gen_config.get('temperature', 0.8),
        top_k=gen_config.get('top_k', 0),
        top_p=gen_config.get('top_p', 0.0)
    )

    generated_text = tokenizer.decode(generated_tokens.tolist())

    print("\n[Шаг 3/3] Результат:")
    print("="*20)
    print(start_text + generated_text)
    print("="*20)

if __name__ == "__main__":
    main()
