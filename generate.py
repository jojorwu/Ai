import numpy as np
import os
import json
from model import Transformer
from tokenizer import Tokenizer

def main():
    """
    Скрипт для генерации текста с использованием двухэтапного процесса "думай, затем отвечай".
    """
    print("--- Запуск генерации текста в режиме 'Думай, затем отвечай' ---")

    # --- 1. Загрузка конфигурации и инициализация ---
    with open('config.json', 'r') as f:
        config = json.load(f)

    train_config = config['training']
    gen_config = config['generation']
    weights_path = train_config['weights_path']
    data_dir = train_config['data_dir']

    print("\n[Шаг 1/4] Загрузка токенизатора и модели...")
    if not os.path.exists(weights_path):
        print(f"Ошибка: Файл с весами '{weights_path}' не найден.")
        return

    tokenizer = Tokenizer(data_dir)
    vocab_size = tokenizer.vocab_size
    model, _ = Transformer.load_model(weights_path, vocab_size)
    model.eval()

    start_text = gen_config['start_text']
    think_token = tokenizer.encode('<THINK>', add_special_tokens=False)
    answer_token = tokenizer.encode('<ANSWER>', add_special_tokens=False)[0]

    # --- 2. Этап "Мышления" ---
    print(f"\n[Шаг 2/4] Генерация внутреннего монолога (мыслей)...")

    # Формируем промпт для мышления: <THINK> + ваш_текст
    prompt_for_thinking = think_token + tokenizer.encode(start_text)

    # Генерируем мысли, пока не встретим токен <ANSWER>
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

    thought_text = tokenizer.decode(thought_tokens)
    print(f"Сгенерированные мысли: {thought_text}")

    # --- 3. Этап "Ответа" ---
    print(f"\n[Шаг 3/4] Генерация финального ответа...")

    # Формируем полный контекст: <THINK> + ваш_текст + мысли + <ANSWER>
    context_for_answer = think_token + tokenizer.encode(start_text) + thought_tokens + [answer_token]

    # Генерируем финальный ответ, используя стандартную генерацию
    final_answer_tokens = model.generate(
        context_for_answer,
        max_len=gen_config.get('max_len', 50),
        temperature=gen_config.get('temperature', 0.8),
        top_k=gen_config.get('top_k', 0),
        top_p=gen_config.get('top_p', 0.0),
        speculative_steps=0 # Отключаем спекулятивную генерацию для чистого ответа
    )

    final_answer_text = tokenizer.decode(final_answer_tokens.tolist())

    # --- 4. Результат ---
    print("\n[Шаг 4/4] Результат:")
    print("="*20)
    print(f"Входной текст: {start_text}")
    print(f"Финальный ответ: {final_answer_text}")
    print("="*20)


if __name__ == "__main__":
    main()
