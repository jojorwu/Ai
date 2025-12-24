import os
import re

class Tokenizer:
    """
    Простой символьный токенизатор с улучшенной поддержкой специальных токенов.
    """
    def __init__(self, data_dir):
        # Добавляем новые специальные токены для использования инструментов
        self.special_tokens = [
            '<THINK>', '<ANSWER>',
            '<TOOL_CALL>', '</TOOL_CALL>',
            '<TOOL_OUTPUT>', '</TOOL_OUTPUT>',
            '<ASK_FOR_HELP>',
            '<I_DONT_KNOW>'
        ]
        self.chars = []
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

        # Создаем регулярное выражение для поиска специальных токенов
        self.special_token_pattern = re.compile(f"({'|'.join(re.escape(token) for token in self.special_tokens)})")

        self._build_vocab(data_dir)

    def _build_vocab(self, data_dir):
        """Строит словарь из всех .txt файлов в директории и добавляет специальные токены."""
        all_text = ""
        # Сканируем директорию для построения словаря символов
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path):
                 # Пропускаем файлы, которые не являются текстовыми
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        all_text += f.read()
                except UnicodeDecodeError:
                    print(f"Skipping non-text file: {filename}")
                    continue

        self.chars = sorted(list(set(all_text)))

        # Добавляем специальные токены в словарь
        full_vocab = self.special_tokens + self.chars
        self.vocab_size = len(full_vocab)

        for i, char in enumerate(full_vocab):
            self.char_to_idx[char] = i
            self.idx_to_char[i] = char

    def encode(self, text: str, add_special_tokens=False) -> list[int]:
        """
        Преобразует строку текста в список токенов, корректно обрабатывая
        специальные токены внутри строки.
        """
        if add_special_tokens:
            text = f"<THINK>{text}<ANSWER>"

        tokens = []
        # Разбиваем текст по специальным токенам
        parts = self.special_token_pattern.split(text)

        for part in parts:
            if not part:
                continue
            # Если часть является специальным токеном, добавляем ее ID
            if part in self.special_tokens:
                tokens.append(self.char_to_idx[part])
            # Иначе, токенизируем ее как обычные символы
            else:
                tokens.extend([self.char_to_idx.get(char, -1) for char in part if char in self.char_to_idx])

        return tokens

    def decode(self, tokens: list[int]) -> str:
        """
        Преобразует список токенов обратно в строку.
        Специальные токены остаются в строке, что важно для контекста модели.
        """
        return "".join([self.idx_to_char.get(token, '') for token in tokens])
