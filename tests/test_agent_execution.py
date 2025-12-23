"""
Тест для проверки основного цикла выполнения агента.
"""
import unittest
import json
import io
from unittest.mock import MagicMock, patch
import sys
import os

# Добавляем корневую директорию проекта в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate import parse_tool_call, main as agent_main

class TestAgentExecution(unittest.TestCase):
    """
    Тестирует основной цикл агента.
    """
    @patch('generate.load_model_and_tokenizer')
    def test_agent_loop_parses_and_executes_tool(self, mock_load_model_and_tokenizer):
        """
        Проверяет, что агент правильно парсит, выполняет вызов инструмента
        и завершает работу после получения ответа.
        """
        # 1. Готовим моки
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # 2. Определяем поведение моков для двух итераций
        # Итерация 1: Модель вызывает инструмент
        tool_call_text = 'Я думаю, надо проверить файлы. <TOOL_CALL>{"tool": "list_files", "args": {"path": "."}}</TOOL_CALL>'
        # Итерация 2: Модель дает финальный ответ
        final_answer_text = 'В директории есть файлы: ["file1.txt", "file2.txt"]. Задача выполнена.'

        # `generate` будет возвращать разные значения при каждом вызове
        mock_model.generate.side_effect = [
            ([1, 2, 3], 0.5, 0.9),  # Фиктивные токены для первого ответа
            ([4, 5, 6], 0.2, 0.8)   # Фиктивные токены для второго ответа
        ]
        # `decode` также будет возвращать разные значения
        mock_tokenizer.decode.side_effect = [
            tool_call_text,
            final_answer_text
        ]
        # `encode` возвращает фиктивные токены
        mock_tokenizer.encode.return_value = [7, 8, 9]

        mock_load_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)

        # 3. Выполняем `agent_main` и проверяем вызовы
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            with patch('generate.execute_tool') as mock_execute_tool:
                mock_execute_tool.return_value = '["file1.txt", "file2.txt"]'

                agent_main()

                # 4. Проверяем, что инструмент был вызван ровно один раз
                mock_execute_tool.assert_called_once_with("list_files", {"path": "."})

                # 5. Проверяем, что финальный ответ был выведен
                self.assertIn(final_answer_text, mock_stdout.getvalue())

        print("Тест test_agent_loop_parses_and_executes_tool PASSED")

    def test_parse_tool_call(self):
        """
        Проверяет, что функция `parse_tool_call` правильно извлекает
        имя инструмента и его аргументы.
        """
        text = 'Вот результат: <TOOL_CALL>{"tool": "write_file", "args": {"path": "out.txt", "content": "hello"}}</TOOL_CALL>'
        tool_name, args = parse_tool_call(text)
        self.assertEqual(tool_name, "write_file")
        self.assertEqual(args, {"path": "out.txt", "content": "hello"})
        print("Тест test_parse_tool_call PASSED")


if __name__ == "__main__":
    unittest.main()
