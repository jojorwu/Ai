import numpy as np
import os
import json
from model import Transformer
from tokenizer import Tokenizer

def load_model_and_tokenizer(weights_path, data_dir):
    """Загружает модель и токенизатор."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Файл с весами '{weights_path}' не найден.")

    tokenizer = Tokenizer(data_dir)
    vocab_size = tokenizer.vocab_size
    model, _ = Transformer.load_model(weights_path, vocab_size)
    model.eval()
    return model, tokenizer

def generate_thought(model, tokenizer, start_text, gen_config):
    """Генерирует внутренний монолог (мысли)."""
    think_token = tokenizer.encode('<THINK>', add_special_tokens=False)
    answer_token = tokenizer.encode('<ANSWER>', add_special_tokens=False)[0]

    prompt_for_thinking = think_token + tokenizer.encode(start_text)

    thought_tokens = []
    generated_tokens_stream = model.generate(
        prompt_for_thinking,
        max_len=gen_config.get('max_thought_len', 50),
        temperature=gen_config.get('temperature', 0.8),
        top_k=gen_config.get('top_k', 0),
        top_p=gen_config.get('top_p', 0.0),
        speculative_steps=gen_config.get('speculative_steps', 5),
        value_threshold=gen_config.get('value_threshold', -1.0),
        max_retries=gen_config.get('max_retries', 3)
    )

    for token in generated_tokens_stream:
        if token == answer_token:
            break
        thought_tokens.append(token)

    return thought_tokens

def generate_answer(model, tokenizer, start_text, thought_tokens, gen_config):
    """Генерирует финальный ответ."""
    think_token = tokenizer.encode('<THINK>', add_special_tokens=False)
    answer_token = tokenizer.encode('<ANSWER>', add_special_tokens=False)[0]

    context_for_answer = think_token + tokenizer.encode(start_text) + thought_tokens + [answer_token]

    final_answer_tokens = model.generate(
        context_for_answer,
        max_len=gen_config.get('max_len', 50),
        temperature=gen_config.get('temperature', 0.8),
        top_k=gen_config.get('top_k', 0),
        top_p=gen_config.get('top_p', 0.0),
        speculative_steps=0
    )

    return tokenizer.decode(final_answer_tokens.tolist())

def main():
    """
    Основной скрипт для генерации текста.
    """
    print("--- Запуск генерации текста ---")

    with open('config.json', 'r') as f:
        config = json.load(f)

    train_config = config['training']
    gen_config = config['generation']

    print("\n[Шаг 1/4] Загрузка модели и токенизатора...")
    model, tokenizer = load_model_and_tokenizer(train_config['weights_path'], train_config['data_dir'])

    start_text = gen_config['start_text']

    print("\n[Шаг 2/4] Генерация мыслей...")
    thought_tokens = generate_thought(model, tokenizer, start_text, gen_config)
    thought_text = tokenizer.decode(thought_tokens)
    print(f"Сгенерированные мысли: {thought_text}")

    print("\n[Шаг 3/4] Генерация ответа...")
    final_answer_text = generate_answer(model, tokenizer, start_text, thought_tokens, gen_config)

    print("\n[Шаг 4/4] Результат:")
    print("="*20)
    print(f"Входной текст: {start_text}")
    print(f"Финальный ответ: {final_answer_text}")
    print("="*20)

if __name__ == "__main__":
    main()
