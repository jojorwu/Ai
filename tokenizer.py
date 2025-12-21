import os

class Tokenizer:
    """
    Простой символьный токенизатор с поддержкой специальных токенов.
    """
    def __init__(self, data_dir):
        self.special_tokens = ['<THINK>', '<ANSWER>']
        self.chars = []
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

        self._build_vocab(data_dir)

    def _build_vocab(self, data_dir):
        """Строит словарь из всех .txt файлов в директории и добавляет специальные токены."""
        all_text = ""
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                    all_text += f.read()

        self.chars = sorted(list(set(all_text)))

        # Add special tokens to the vocabulary
        full_vocab = self.special_tokens + self.chars
        self.vocab_size = len(full_vocab)

        for i, char in enumerate(full_vocab):
            self.char_to_idx[char] = i
            self.idx_to_char[i] = char

    def encode(self, text, add_special_tokens=False):
        """
        Преобразует строку текста в список токенов.
        Опционально добавляет специальные токены в начало и конец.
        """
        tokens = []
        if add_special_tokens:
            tokens.append(self.char_to_idx['<THINK>'])

        # This is a simplified approach. A more robust tokenizer would handle
        # special tokens within the text itself.
        if text in self.special_tokens:
            return [self.char_to_idx[text]]

        tokens.extend([self.char_to_idx[char] for char in text])

        if add_special_tokens:
            tokens.append(self.char_to_idx['<ANSWER>'])

        return tokens

    def decode(self, tokens):
        """Преобразует список токенов обратно в строку."""
        # Filter out special tokens from the decoded string for clean output
        return "".join([self.idx_to_char[token] for token in tokens if self.idx_to_char[token] not in self.special_tokens])
