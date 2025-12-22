"""
Скрипт для генерации текста с использованием обученной модели Трансформер.
Поддерживает генерацию "мыслей" (внутреннего монолога) перед ответом.
"""

import logging
import os
from copy import deepcopy

from config import Config, GenerationConfig, TrainingConfig
from model import Transformer
from tokenizer import Tokenizer

def setup_logging():
    """Настраивает логирование в консоль."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_tokenizer(train_config: TrainingConfig):
    """
    Загружает модель и токенизатор.
    Включает проверку на наличие файла с весами.
    """
    if not os.path.exists(train_config.weights_path):
        logging.error("Файл с весами '%s' не найден.", train_config.weights_path)
        raise FileNotFoundError(f"Файл с весами '{train_config.weights_path}' не найден.")

    tokenizer = Tokenizer(train_config.data_dir)
    model, _ = Transformer.load_model(train_config.weights_path, tokenizer.vocab_size)
    model.eval()
    return model, tokenizer

def generate_text_stream(model: Transformer, prompt: list, gen_config: GenerationConfig, stop_token: int = None):
    """
    Универсальная функция для потоковой генерации токенов.
    """
    generation_params = gen_config.dict()
    # Удаляем start_text, так как он не является параметром для model.generate
    generation_params.pop('start_text', None)
    generation_params.pop('max_thought_len', None)

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
        config = Config.from_json('config.json')
        train_config = config.training
        gen_config = config.generation

        logging.info("[Шаг 1/3] Загрузка модели и токенизатора...")
        model, tokenizer = load_model_and_tokenizer(train_config)

        logging.info("Входной текст: %s", gen_config.start_text)

        # Подготовка специальных токенов
        think_token_id = tokenizer.encode('<THINK>', add_special_tokens=False)
        answer_token_id = tokenizer.encode('<ANSWER>', add_special_tokens=False)[0]

        # Генерация "мыслей"
        logging.info("[Шаг 2/3] Генерация мыслей (внутреннего монолога)...")
        prompt_for_thinking = think_token_id + tokenizer.encode(gen_config.start_text)

        thought_gen_config = deepcopy(gen_config)
        thought_gen_config.max_len = thought_gen_config.max_thought_len

        thought_tokens = generate_text_stream(model, prompt_for_thinking, thought_gen_config, stop_token=answer_token_id)
        thought_text = tokenizer.decode(thought_tokens)
        logging.info("Сгенерированные мысли: %s", thought_text)

        # Генерация ответа
        logging.info("[Шаг 3/3] Генерация финального ответа...")
        prompt_for_answer = prompt_for_thinking + thought_tokens + [answer_token_id]

        answer_gen_config = deepcopy(gen_config)
        answer_gen_config.speculative_steps = 0  # Ответ генерируем без спекуляции

        answer_tokens = generate_text_stream(model, prompt_for_answer, answer_gen_config)
        final_answer_text = tokenizer.decode(answer_tokens)

        print("\\n" + "="*30)
        print(f"  Входной текст: {gen_config.start_text}")
        print(f"  Финальный ответ: {final_answer_text}")
        print("="*30)

    except FileNotFoundError as e:
        logging.error("Ошибка: %s. Убедитесь, что модель обучена и файл 'config.json' настроен правильно.", e)
    except Exception as e:
        logging.error("Произошла непредвиденная ошибка: %s", e, exc_info=True)

if __name__ == "__main__":
    main()
