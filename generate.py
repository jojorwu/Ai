"""
Основной скрипт-агент для взаимодействия с моделью Трансформер.
Этот скрипт управляет циклом "мысль -> инструмент -> наблюдение",
позволяя модели использовать инструменты для выполнения задач.
"""

import json
import logging
import re
from copy import deepcopy

from config import Config, TrainingConfig
from model import Transformer
from tokenizer import Tokenizer
from tools import execute_tool

def setup_logging():
    """Настраивает логирование в консоль."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_model_and_tokenizer(config: Config):
    """Загружает модель и токенизатор."""
    logging.info("Загрузка модели и токенизатора...")
    tokenizer = Tokenizer(config.training.data_dir)
    # Теперь load_model принимает весь конфиг
    model = Transformer.load_model(config.training.weights_path, tokenizer.vocab_size, config)
    model.eval()
    logging.info("Модель и токенизатор успешно загружены.")
    return model, tokenizer

def parse_tool_call(text: str) -> tuple[str | None, dict | None]:
    """
    Ищет в тексте вызов инструмента, заключенный в <TOOL_CALL>...</TOOL_CALL>,
    и парсит его.
    """
    pattern = r"<TOOL_CALL>(.*?)</TOOL_CALL>"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None, None

    tool_call_json = match.group(1).strip()
    try:
        tool_call = json.loads(tool_call_json)
        tool_name = tool_call.get("tool")
        args = tool_call.get("args", {})
        if not isinstance(tool_name, str) or not isinstance(args, dict):
            return None, None
        return tool_name, args
    except (json.JSONDecodeError, AttributeError):
        return None, None

from backend import set_backend

def main():
    """
    Основной цикл агента.
    """
    setup_logging()

    try:
        # Загрузка конфигурации и установка бэкенда
        config = Config.from_json('config.json')
        set_backend(config.hardware.device)

        model, tokenizer = load_model_and_tokenizer(config)

        # Начальная инструкция для модели
        start_text = config.generation.start_text
        logging.info(f"Начальная задача: {start_text}")

        # Формируем начальный промпт
        conversation_history_tokens = tokenizer.encode(f"<THINK>{start_text}")

        for turn in range(config.generation.max_turns):
            logging.info(f"\n--- Итерация {turn + 1} ---")

            # --- Генерация ответа модели ---
            gen_config = deepcopy(config.generation)
            gen_config.speculative_steps = 0 # Отключаем спекуляцию для более точных вызовов

            # Генерируем продолжение диалога
            generated_tokens_stream, _, _ = model.generate(
                conversation_history_tokens,
                **gen_config.dict()
            )

            # Декодируем сгенерированный текст
            generated_text = tokenizer.decode(list(generated_tokens_stream))
            logging.info(f"Модель сгенерировала:\n{generated_text}")

            # Добавляем сгенерированные токены в историю
            conversation_history_tokens.extend(list(generated_tokens_stream))

            # --- Поиск и выполнение вызова инструмента ---
            tool_name, args = parse_tool_call(generated_text)

            if tool_name and args is not None:
                tool_output = execute_tool(tool_name, args)
                logging.info(f"Вывод инструмента '{tool_name}':\n{tool_output}")

                # Формируем и добавляем результат работы инструмента в историю
                tool_output_formatted = f"<TOOL_OUTPUT>{tool_output}</TOOL_OUTPUT>"
                tool_output_tokens = tokenizer.encode(tool_output_formatted)
                conversation_history_tokens.extend(tool_output_tokens)
            else:
                # Если вызова инструмента нет, считаем, что модель дала финальный ответ
                logging.info("\n--- Финальный ответ ---")
                final_answer = generated_text.split("</TOOL_CALL>")[-1].strip()
                print(final_answer)
                break
        else:
            logging.warning("Достигнуто максимальное количество итераций. Завершение работы.")

    except FileNotFoundError as e:
        logging.error(f"Ошибка: {e}. Убедитесь, что модель обучена и 'config.json' настроен.")
    except Exception as e:
        logging.error(f"Произошла непредвиденная ошибка: {e}", exc_info=True)

if __name__ == "__main__":
    main()
