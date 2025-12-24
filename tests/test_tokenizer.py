"""
Tests for the updated Tokenizer.
"""
import os
import sys
import shutil
import unittest

# Добавляем корневую директорию проекта в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):
    """
    Тесты для токенизатора с улучшенной обработкой специальных токенов.
    """
    def setUp(self):
        """Настройка тестового окружения."""
        self.test_dir = "test_data_tokenizer"
        os.makedirs(self.test_dir, exist_ok=True)
        # Создаем тестовый файл с символами, необходимыми для всех тестов
        with open(os.path.join(self.test_dir, "a.txt"), "w", encoding="utf-8") as f:
            f.write("abcstart{{\"tool\": \"test\"}}end")
        self.tokenizer = Tokenizer(self.test_dir)

    def tearDown(self):
        """Очистка после тестов."""
        shutil.rmtree(self.test_dir)

    def test_vocab_size_includes_all_special_tokens(self):
        """
        Проверяет, что размер словаря правильный и включает все
        специальные токены и уникальные символы.
        """
        # 8 спец. токенов + уникальные символы из файла
        unique_chars = set("abcstart{{\"tool\": \"test\"}}end")
        expected_vocab_size = 8 + len(unique_chars)
        self.assertEqual(self.tokenizer.vocab_size, expected_vocab_size)
        self.assertIn('<TOOL_CALL>', self.tokenizer.char_to_idx)
        self.assertIn('{', self.tokenizer.char_to_idx)
        print("Тест test_vocab_size_includes_all_special_tokens PASSED")

    def test_encode_handles_inline_special_tokens(self):
        """
        Проверяет, что `encode` корректно обрабатывает строку,
        содержащую специальные токены.
        """
        text = "a<TOOL_CALL>b</TOOL_CALL>c"
        encoded = self.tokenizer.encode(text)

        expected_tokens = [
            self.tokenizer.char_to_idx['a'],
            self.tokenizer.char_to_idx['<TOOL_CALL>'],
            self.tokenizer.char_to_idx['b'],
            self.tokenizer.char_to_idx['</TOOL_CALL>'],
            self.tokenizer.char_to_idx['c']
        ]

        self.assertEqual(encoded, expected_tokens)
        print("Тест test_encode_handles_inline_special_tokens PASSED")

    def test_decode_preserves_special_tokens(self):
        """
        Проверяет, что `decode` не удаляет, а сохраняет специальные
        токены в результирующей строке.
        """
        tokens = [
            self.tokenizer.char_to_idx['<THINK>'],
            self.tokenizer.char_to_idx['a'],
            self.tokenizer.char_to_idx['b'],
            self.tokenizer.char_to_idx['<ANSWER>']
        ]
        decoded = self.tokenizer.decode(tokens)
        self.assertEqual(decoded, "<THINK>ab<ANSWER>")
        print("Тест test_decode_preserves_special_tokens PASSED")

    def test_encode_decode_is_reversible(self):
        """
        Проверяет, что операция encode -> decode является обратимой.
        """
        original_text = "start<TOOL_CALL>{\"tool\":\"test\"}</TOOL_CALL>end"
        encoded = self.tokenizer.encode(original_text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(original_text, decoded)
        print("Тест test_encode_decode_is_reversible PASSED")


if __name__ == "__main__":
    unittest.main()
