"""
Скрипт для генерации текста с использованием обученной модели Трансформер.
Поддерживает генерацию "мыслей" (внутреннего монолога) перед ответом.
"""

import json
import logging
import os
import numpy as np

from model import Transformer
from tokenizer import Tokenizer

def setup_logging():
    """Настраивает логирование в консоль."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path='config.json'):
    """Загружает конфигурацию из JSON файла."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_model_and_tokenizer(weights_path, data_dir):
    """
    Загружает модель и токенизатор.
    Включает проверку на наличие файла с весами.
    """
    if not os.path.exists(weights_path):
        logging.error("Файл с весами '%s' не найден.", weights_path)
        raise FileNotFoundError(f"Файл с весами '{weights_path}' не найден.")

    tokenizer = Tokenizer(data_dir)
    model, _ = Transformer.load_model(weights_path, tokenizer.vocab_size)
    model.eval()
    return model, tokenizer

def generate_text_stream(model, prompt, gen_config, stop_token=None):
    """
    Универсальная функция для потоковой генерации токенов.
    """
    generation_params = {
        'max_len': gen_config.get('max_len', 50),
        'temperature': gen_config.get('temperature', 0.8),
        'top_k': gen_config.get('top_k', 0),
        'top_p': gen_config.get('top_p', 0.0),
        'speculative_steps': gen_config.get('speculative_steps', 5),
        'value_threshold': gen_config.get('value_threshold', -1.0),
        'max_retries': gen_config.get('max_retries', 3)
    }

    generated_tokens = []
    token_stream = model.generate(prompt, **generation_params)

    for token in token_stream:
        if stop_token is not None and token == stop_token:
            break
        generated_tokens.append(token)

    return generated_tokens

def main():
    """
    Основной скрипт для генерации текста.
    """
    setup_logging()
    logging.info("--- Запуск генерации текста ---")

    try:
        config = load_config()
        train_config = config['training']
        gen_config = config['generation']

        logging.info("[Шаг 1/3] Загрузка модели и токенизатора...")
        model, tokenizer = load_model_and_tokenizer(train_config['weights_path'], train_config['data_dir'])

        start_text = gen_config['start_text']
        logging.info("Входной текст: %s", start_text)

        # Подготовка специальных токенов
        think_token_id = tokenizer.encode('<THINK>', add_special_tokens=False)
        answer_token_id = tokenizer.encode('<ANSWER>', add_special_tokens=False)[0]

        # Генерация "мыслей"
        logging.info("[Шаг 2/3] Генерация мыслей (внутреннего монолога)...")
        prompt_for_thinking = think_token_id + tokenizer.encode(start_text)

        thought_gen_config = gen_config.copy()
        thought_gen_config['max_len'] = thought_gen_config.get('max_thought_len', 50)

        thought_tokens = generate_text_stream(model, prompt_for_thinking, thought_gen_config, stop_token=answer_token_id)
        thought_text = tokenizer.decode(thought_tokens)
        logging.info("Сгенерированные мысли: %s", thought_text)

        # Генерация ответа
        logging.info("[Ша-г 3/3] Генерация финального ответа...")
        prompt_for_answer = prompt_for_thinking + thought_tokens + [answer_token_id]

        answer_gen_config = gen_config.copy()
        answer_gen_config['speculative_steps'] = 0 # Ответ генерируем без спекуляции

        answer_tokens = generate_text_stream(model, prompt_for_answer, answer_gen_config)
        final_answer_text = tokenizer.decode(answer_tokens)

        print("\\n" + "="*30)
        print(f"  Входной текст: {start_text}")
        print(f"  Финальный ответ: {final_answer_text}")
        print("="*30)

    except FileNotFoundError as e:
        logging.error("Ошибка: %s. Убедитесь, что модель обучена и файл 'config.json' настроен правильно.", e)
    except Exception as e:
        logging.error("Произошла непредвиденная ошибка: %s", e, exc_info=True)

if __name__ == "__main__":
    main()
